from abc import ABC, abstractmethod
import numpy as np
import mici.autograd_wrapper as autograd_wrapper
from mici.states import cache_in_state, cache_in_state_with_aux
import mici.matrices as matrices
from mici.errors import NonReversibleStepError, AdaptationError
from mici.errors import IntegratorError
from mici.errors import ConvergenceError, LinAlgError
import tempfile
from mici.progressbars import (
    ProgressBar,
    LabelledSequenceProgressBar,
    DummyProgressBar,
    _ProxyProgressBar,
)
from mici.states import ChainState
import os
import logging
import sys
import inspect
from numpy.random import default_rng
from contextlib import ExitStack, contextmanager
import signal
import queue
from pickle import PicklingError
from mici.stagers import WarmUpStager, WindowedWarmUpStager
from warnings import warn
import mici.transitions as trans
from collections import OrderedDict
#from mici.adapters import DualAveragingStepSizeAdapter
from math import log, exp


try:
    from multiprocess import Pool
    from multiprocess.managers import SyncManager

    MULTIPROCESS_AVAILABLE = True
except ImportError:
    from multiprocessing import Pool
    from multiprocessing.managers import SyncManager

    MULTIPROCESS_AVAILABLE = False

try:
    from threadpoolctl import threadpool_limits

    THREADPOOLCTL_AVAILABLE = True
except ImportError:
    THREADPOOLCTL_AVAILABLE = False

try:
    from contextlib import nullcontext
except ImportError:
    # Fallback for nullcontext context manager for Python 3.6
    # https://stackoverflow.com/a/55902915
    @contextmanager
    def nullcontext():
        yield None


logger = logging.getLogger(__name__)

"""List of names of valid differential operators.

Any automatic differentiation framework wrapper module will need to provide
all of these operators as callables (with a single function as argument) to
fully support all of the required derivative functions.
"""
DIFF_OPS = [
    # vector Jacobian product and value
    "vjp_and_value",
    # gradient and value for scalar valued functions
    "grad_and_value",
    # Hessian matrix, gradient and value for scalar valued functions
    "hessian_grad_and_value",
    # matrix Tressian product, gradient and value for scalar valued
    # functions
    "mtp_hessian_grad_and_value",
    # Jacobian matrix and value for vector valued functions
    "jacobian_and_value",
    # matrix Hessian product, Jacobian matrix and value for vector valued
    # functions
    "mhp_jacobian_and_value",
]

def autodiff_fallback(diff_func, func, diff_op_name, name):
    """Generate derivative function automatically if not provided.

    Uses automatic differentiation to generate a function corresponding to a
    differential operator applied to a function if an alternative
    implementation of the derivative function has not been provided.

    Args:
        diff_func (None or Callable): Either a callable implementing the
            required derivative function or `None` if none was provided.
        func (Callable): Function to differentiate.
        diff_op_name (str): String specifying name of differential operator
            from automatic differentiation framework wrapper to use to generate
            required derivative function.
        name (str): Name of derivative function to use in error message.

    Returns:
        Callable: `diff_func` value if not `None` otherwise generated
            derivative of `func` by applying named differential operator.
    """
    if diff_func is not None:
        return diff_func
    elif diff_op_name not in DIFF_OPS:
        raise ValueError(f"Differential operator {diff_op_name} is not defined.")
    elif autograd_wrapper.AUTOGRAD_AVAILABLE:
        return getattr(autograd_wrapper, diff_op_name)(func)
    elif not autograd_wrapper.AUTOGRAD_AVAILABLE:
        raise ValueError(f"Autograd not available therefore {name} must be provided.")


class System(ABC):
    r"""Base class for Hamiltonian systems.

    The Hamiltonian function \(h\) is assumed to have the general form

    \[ h(q, p) = h_1(q) + h_2(q, p) \]

    where \(q\) and \(p\) are the position and momentum variables respectively,
    and \(h_1\) and \(h_2\) Hamiltonian component functions. The exact
    Hamiltonian flow for the \(h_1\) component can be always be computed as it
    depends only on the position variable however depending on the form of
    \(h_2\) the corresponding exact Hamiltonian flow may or may not be
    simulable.

    By default \(h_1\) is assumed to correspond to the negative logarithm of an
    unnormalized density on the position variables with respect to the Lebesgue
    measure, with the corresponding distribution on the position space being
    the target distribution it is wished to draw approximate samples from.
    """

    def __init__(self, neg_log_dens, grad_neg_log_dens=None):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalized probability density on the position space with
                respect to the Lebesgue measure, with the corresponding
                distribution on the position space being the target
                distribution it is wished to draw approximate samples from.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct the derivative of `neg_log_dens` automatically.
        """
        self._neg_log_dens = neg_log_dens
        self._grad_neg_log_dens = autodiff_fallback(
            grad_neg_log_dens, neg_log_dens, "grad_and_value", "grad_neg_log_dens"
        )

    @cache_in_state("pos")
    def neg_log_dens(self, state):
        """Negative logarithm of unnormalized density of target distribution.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            float: Value of computed negative log density.
        """
        return self._neg_log_dens(state.pos)

    @cache_in_state_with_aux("pos", "neg_log_dens")
    def grad_neg_log_dens(self, state):
        """Derivative of negative log density with respect to position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of `neg_log_dens(state)` derivative with respect to
                `state.pos`.
        """
        return self._grad_neg_log_dens(state.pos)

    def h1(self, state):
        """Hamiltonian component depending only on position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            float: Value of `h1` Hamiltonian component.
        """
        return self.neg_log_dens(state)

    def dh1_dpos(self, state):
        """Derivative of `h1` Hamiltonian component with respect to position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of computed `h1` derivative.
        """
        return self.grad_neg_log_dens(state)

    def h1_flow(self, state, dt):
        """Apply exact flow map corresponding to `h1` Hamiltonian component.

        `state` argument is modified in place.

        Args:
            state (mici.states.ChainState): State to start flow at.
            dt (float): Time interval to simulate flow for.
        """
        state.mom -= dt * self.dh1_dpos(state)

    @abstractmethod
    def h2(self, state):
        """Hamiltonian component depending on momentum and optionally position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            float: Value of `h2` Hamiltonian component.
        """

    @abstractmethod
    def dh2_dmom(self, state):
        """Derivative of `h2` Hamiltonian component with respect to momentum.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of `h2(state)` derivative with respect to `state.pos`.
        """

    def h(self, state):
        """Hamiltonian function for system.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            float: Value of Hamiltonian.
        """
        return self.h1(state) + self.h2(state)

    def dh_dpos(self, state):
        """Derivative of Hamiltonian with respect to position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of `h(state)` derivative with respect to `state.pos`.
        """
        if hasattr(self, "dh2_dpos"):
            return self.dh1_dpos(state) + self.dh2_dpos(state)
        else:
            return self.dh1_dpos(state)

    def dh_dmom(self, state):
        """Derivative of Hamiltonian with respect to momentum.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of `h(state)` derivative with respect to `state.mom`.
        """
        return self.dh2_dmom(state)

    @abstractmethod
    def sample_momentum(self, state, rng):
        """
        Sample a momentum from its conditional distribution given a position.

        Args:
            state (mici.states.ChainState): State defining position to
               condition on.

        Returns:
            mom (array): Sampled momentum.
        """


class EuclideanMetricSystem(System):
    r"""Hamiltonian system with a Euclidean metric on the position space.

    Here Euclidean metric is defined to mean a metric with a fixed positive
    definite matrix representation \(M\). The momentum variables are taken to
    be independent of the position variables and with a zero-mean Gaussian
    marginal distribution with covariance specified by \(M\), so that the
    \(h_2\) Hamiltonian component is

    \[ h_2(q, p) = \frac{1}{2} p^T M^{-1} p \]

    where \(q\) and \(p\) are the position and momentum variables respectively.

    The \(h_1\) Hamiltonian component function is

    \[ h_1(q) = \ell(q) \]

    where \(\ell(q)\) is the negative log (unnormalized) density of
    the target distribution with respect to the Lebesgue measure.
    """

    def __init__(self, neg_log_dens, metric=None, grad_neg_log_dens=None):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalized probability density on the position space with
                respect to the Lebesgue measure, with the corresponding
                distribution on the position space being the target
                distribution it is wished to draw approximate samples from.
            metric (None or array or PositiveDefiniteMatrix): Matrix object
                corresponding to matrix representation of metric on position
                space and covariance of Gaussian marginal distribution on
                momentum vector. If `None` is passed (the default), the
                identity matrix will be used. If a 1D array is passed then this
                is assumed to specify a metric with positive diagonal matrix
                representation and the array the matrix diagonal. If a 2D array
                is passed then this is assumed to specify a metric with a dense
                positive definite matrix representation specified by the array.
                Otherwise if the value is a subclass of
                `mici.matrices.PositiveDefiniteMatrix` it is assumed to
                directly specify the metric matrix representation.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct the derivative of `neg_log_dens` automatically.
        """
        super().__init__(neg_log_dens, grad_neg_log_dens)
        if metric is None:
            self.metric = matrices.IdentityMatrix()
        elif isinstance(metric, np.ndarray):
            if metric.ndim == 1:
                self.metric = matrices.PositiveDiagonalMatrix(metric)
            elif metric.ndim == 2:
                self.metric = matrices.DensePositiveDefiniteMatrix(metric)
            else:
                raise ValueError(
                    "If NumPy ndarray value is used for `metric`"
                    " must be either 1D (diagonal matrix) or 2D "
                    "(dense positive definite matrix)"
                )
        else:
            self.metric = metric

    @cache_in_state("mom")
    def h2(self, state):
        return 0.5 * state.mom @ self.dh2_dmom(state)

    @cache_in_state("mom")
    def dh2_dmom(self, state):
        return self.metric.inv @ state.mom

    def h2_flow(self, state, dt):
        """Apply exact flow map corresponding to `h2` Hamiltonian component.

        `state` argument is modified in place.

        Args:
            state (mici.states.ChainState): State to start flow at.
            dt (float): Time interval to simulate flow for.
        """
        state.pos += dt * self.dh2_dmom(state)

    def dh2_flow_dmom(self, dt):
        """Derivatives of `h2_flow` flow map with respect to input momentum.

        Args:
            dt (float): Time interval flow simulated for.

        Returns:
            dpos_dmom (mici.matrices.Matrix): Matrix representing derivative
                (Jacobian) of position output of `h2_flow` with respect to the
                value of the momentum component of the initial input state.
            dmom_dmom (mici.matrices.Matrix): Matrix representing derivative
                (Jacobian) of momentum output of `h2_flow` with respect to the
                value of the momentum component of the initial input state.
        """
        return (dt * self.metric.inv, matrices.IdentityMatrix(self.metric.shape[0]))

    def sample_momentum(self, state, rng):
        return self.metric.sqrt @ rng.standard_normal(state.pos.shape)


class ConstrainedEuclideanMetricSystem(EuclideanMetricSystem):
    r"""Base class for Euclidean Hamiltonian systems subject to constraints.

    The (constrained) position space is assumed to be a differentiable manifold
    embedded with a \(Q\)-dimensional ambient Euclidean space. The \(Q-C\)
    dimensional manifold \(\mathcal{M}\) is implicitly defined by an equation
    \(\mathcal{M} = \lbrace q \in \mathbb{R}^Q : c(q) = 0 \rbrace\) with
    \(c: \mathbb{R}^Q \to \mathbb{R}^C\) the *constraint function*.

    The ambient Euclidean space is assumed to be equipped with a metric with
    constant positive-definite matrix representation \(M\) which further
    specifies the covariance of the zero-mean Gaussian distribution
    \(\mathcal{N}(0, M)\) on the *unconstrained* momentum (co-)vector \(p\)
    with corresponding \(h_2\) Hamiltonian component defined as

    \[ h_2(q, p) = \frac{1}{2} p^T M^{-1} p. \]

    The time-derivative of the constraint equation implies a further set of
    constraints on the momentum \(q\) with \( \partial c(q) M^{-1} p = 0\)
    at all time points, corresponding to the momentum (velocity) being in the
    co-tangent space (tangent space) to the manifold.

    The target distribution is either assumed to be directly specified with
    unnormalized density \(\exp(-\ell(q))\) with respect to the Hausdorff
    measure on the manifold (under the metric induced from the ambient metric)
    with in this case the \(h_1\) Hamiltonian component then simply

    \[ h_1(q) = \ell(q), \]

    or alternatively it is assumed a prior distribution on the position \(q\)
    with density \(\exp(-\ell(q))\) with respect to the Lebesgue measure on
    the ambient space is specifed and the target distribution is the posterior
    distribution on \(q\) when conditioning on the event \(c(q) = 0\). The
    negative logarithm of the posterior distribution density with respect to
    the Hausdorff measure (and so \(h_1\) Hamiltonian component) is then

    \[
      h_1(q) =
      \ell(q) + \frac{1}{2} \log\left|\partial c(q)M^{-1}\partial c(q)^T\right|
    \]

    with an additional second *Gram matrix* determinant term to give the
    correct density with respect to the Hausdorff measure on the manifold.

    Due to the requirement to enforce the constraints on the position and
    momentum, a constraint-preserving numerical integrator needs to be used
    when simulating the Hamiltonian dynamic associated with the system, e.g.
    `mici.integrators.ConstrainedLeapfrogIntegrator`.

    References:

      1. Lelièvre, T., Rousset, M. and Stoltz, G., 2019. Hybrid Monte Carlo
         methods for sampling probability measures on submanifolds. Numerische
         Mathematik, 143(2), pp.379-421.
      2. Graham, M.M. and Storkey, A.J., 2017. Asymptotically exact inference
         in differentiable generative models. Electronic Journal of Statistics,
         11(2), pp.5105-5164.
    """

    def __init__(
        self,
        neg_log_dens,
        constr,
        metric=None,
        dens_wrt_hausdorff=True,
        grad_neg_log_dens=None,
        jacob_constr=None,
    ):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalized probability density on the constrained position
                space with respect to the Hausdorff measure on the constraint
                manifold (if `dens_wrt_hausdorff == True`) or alternatively the
                negative logarithm of an unnormalized probability density on
                the unconstrained (ambient) position space with respect to the
                Lebesgue measure. In the former case the target distribution it
                is wished to draw approximate samples from is assumed to be
                directly specified by the density function on the manifold. In
                the latter case the density function is instead taken to
                specify a prior distribution on the ambient space with the
                target distribution then corresponding to the posterior
                distribution when conditioning on the (zero Lebesgue measure)
                event `constr(pos) == 0`. This target posterior distribution
                has support on the differentiable manifold implicitly defined
                by the constraint equation, with density with respect to the
                Hausdorff measure on the manifold corresponding to the ratio of
                the prior density (specified by `neg_log_dens`) and the
                square-root of the determinant of the Gram matrix defined by

                    gram(q) = jacob_constr(q) @ inv(metric) @ jacob_constr(q).T

                where `jacob_constr` is the Jacobian of the constraint function
                `constr` and `metric` is the matrix representation of the
                metric on the ambient space.
            constr (Callable[[array], array]): Function which given a position
                array return as a 1D array the value of the (vector-valued)
                constraint function, the zero level-set of which implicitly
                defines the manifold the dynamic is simulated on.
            metric (None or array or PositiveDefiniteMatrix): Matrix object
                corresponding to matrix representation of metric on
                *unconstrained* position space and covariance of Gaussian
                marginal distribution on *unconstrained* momentum vector. If
                `None` is passed (the default), the identity matrix will be
                used. If a 1D array is passed then this is assumed to specify a
                metric with positive diagonal matrix representation and the
                array the matrix diagonal. If a 2D array is passed then this is
                assumed to specify a metric with a dense positive definite
                matrix representation specified by the array. Otherwise if the
                value is a `mici.matrices.PositiveDefiniteMatrix` subclass it
                is assumed to directly specify the metric matrix
                representation.
            dens_wrt_hausdorff (bool): Whether the `neg_log_dens` function
                specifies the (negative logarithm) of the density of the target
                distribution with respect to the Hausdorff measure on the
                manifold directly (True) or alternatively the negative
                logarithm of a density of a prior distriubtion on the
                unconstrained (ambient) position space with respect to the
                Lebesgue measure, with the target distribution then
                corresponding to the posterior distribution when conditioning
                on the event `const(pos) == 0` (False). Note that in the former
                case the base Hausdorff measure on the manifold depends on the
                metric defined on the ambient space, with the Hausdorff measure
                being defined with respect to the metric induced on the
                manifold from this ambient metric.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct a function to compute the derivative (and value)
                of `neg_log_dens` automatically.
            jacob_constr (
                    None or Callable[[array], array or Tuple[array, array]]):
                Function which given a position array computes the Jacobian
                (matrix / 2D array of partial derivatives) of the output of the
                constraint function `c = constr(q)` with respect to the position
                array argument `q`, returning the computed Jacobian as a 2D
                array `jacob` with

                    jacob[i, j] = ∂c[i] / ∂q[j]

                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the Jacobian and
                the second being the value of `constr` evaluated at the passed
                position array. If `None` is passed (the default) an automatic
                differentiation fallback will be used to attempt to construct a
                function to compute the Jacobian (and value) of `constr`
                automatically.
        """
        super().__init__(
            neg_log_dens=neg_log_dens,
            metric=metric,
            grad_neg_log_dens=grad_neg_log_dens,
        )
        self._constr = constr
        self.dens_wrt_hausdorff = dens_wrt_hausdorff
        self._jacob_constr = autodiff_fallback(
            jacob_constr, constr, "jacobian_and_value", "jacob_constr"
        )

    @cache_in_state("pos")
    def constr(self, state):
        """Constraint function at the current position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of `constr(state.pos)` as 1D array.
        """
        return self._constr(state.pos)

    @cache_in_state_with_aux("pos", "constr")
    def jacob_constr(self, state):
        """Jacobian of constraint function at the current position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of Jacobian of `constr(state.pos)` as 2D array.
        """
        return self._jacob_constr(state.pos)

    @abstractmethod
    def jacob_constr_inner_product(
        self, jacob_constr_1, inner_product_matrix, jacob_constr_2=None
    ):
        """Compute inner product of rows of constraint Jacobian matrices.

        Computes `jacob_constr_1 @ inner_product_matrix @ jacob_constr_2.T`
        potentially exploiting any structure / sparsity in `jacob_constr_1`,
        `jacob_constr_2` and `inner_product_matrix`.

        Args:
            jacob_constr_1 (Matrix): First constraint Jacobian in product.
            inner_product_matrix (Matrix): Positive-definite matrix defining
                inner-product between rows of two constraint Jacobians.
            jacob_constr_2 (None or Matrix): Second constraint Jacobian in
                product. Defaults to `jacob_constr_1` if set to `None`.

        Returns
            Matrix: Object corresponding to computed inner products of
               the constraint Jacobian rows.
        """

    @cache_in_state("pos")
    def gram(self, state):
        """Gram matrix at current position.

        The Gram matrix as a position `q` is defined as

            gram(q) = jacob_constr(q) @ inv(metric) @ jacob_constr(q).T

        where `jacob_constr` is the Jacobian of the constraint function
        `constr` and `metric` is the matrix representation of the metric on the
        ambient space.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            mici.matrices.PositiveDefiniteMatrix: Gram matrix as matrix object.
        """
        return self.jacob_constr_inner_product(
            self.jacob_constr(state), self.metric.inv
        )

    def inv_gram(self, state):
        """Inverse of Gram matrix at current position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            mici.matrices.PositiveDefiniteMatrix: Inverse of Gram matrix as
                matrix object.
        """
        return self.gram(state).inv

    def log_det_sqrt_gram(self, state):
        """Value of (half of) log-determinant of Gram matrix."""
        return 0.5 * self.gram(state).log_abs_det

    @abstractmethod
    def grad_log_det_sqrt_gram(self, state):
        """Derivative of (half of) log-determinant of Gram matrix wrt position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of `log_det_sqrt_gram(state)` derivative with respect
                to `state.pos`.
        """

    def h1(self, state):
        if self.dens_wrt_hausdorff:
            return self.neg_log_dens(state)
        else:
            return self.neg_log_dens(state) + self.log_det_sqrt_gram(state)

    def dh1_dpos(self, state):
        if self.dens_wrt_hausdorff:
            return self.grad_neg_log_dens(state)
        else:
            return self.grad_neg_log_dens(state) + self.grad_log_det_sqrt_gram(state)

    def project_onto_cotangent_space(self, mom, state):
        """Project a momentum on to the co-tangent space at a position.

        Args:
            mom (array): Momentum (co-)vector as 1D array to project on to
                co-tangent space.
            state (mici.states.ChainState): State definining position on the
                manifold to project in to the co-tangent space of.

        Returns:
            array: Projected momentum in the co-tangent space at `state.pos`.
        """
        # Use parenthesis to force right-to-left evaluation to avoid
        # matrix-matrix products
        mom -= self.jacob_constr(state).T @ (
            self.inv_gram(state) @ (self.jacob_constr(state) @ (self.metric.inv @ mom))
        )
        return mom

    def sample_momentum(self, state, rng):
        mom = super().sample_momentum(state, rng)
        mom = self.project_onto_cotangent_space(mom, state)
        return mom


class DenseConstrainedEuclideanMetricSystem(ConstrainedEuclideanMetricSystem):
    r"""Euclidean Hamiltonian system subject to a dense set of constraints.

    See `ConstrainedEuclideanMetricSystem` for more details about constrained
    systems.
    """

    def __init__(
        self,
        neg_log_dens,
        constr,
        metric=None,
        dens_wrt_hausdorff=True,
        grad_neg_log_dens=None,
        jacob_constr=None,
        mhp_constr=None,
    ):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalized probability density on the constrained position
                space with respect to the Hausdorff measure on the constraint
                manifold (if `dens_wrt_hausdorff == True`) or alternatively the
                negative logarithm of an unnormalized probability density on
                the unconstrained (ambient) position space with respect to the
                Lebesgue measure. In the former case the target distribution it
                is wished to draw approximate samples from is assumed to be
                directly specified by the density function on the manifold. In
                the latter case the density function is instead taken to
                specify a prior distribution on the ambient space with the
                target distribution then corresponding to the posterior
                distribution when conditioning on the (zero Lebesgue measure)
                event `constr(pos) == 0`. This target posterior distribution
                has support on the differentiable manifold implicitly defined
                by the constraint equation, with density with respect to the
                Hausdorff measure on the manifold corresponding to the ratio of
                the prior density (specified by `neg_log_dens`) and the
                square-root of the determinant of the Gram matrix defined by

                    gram(q) = jacob_constr(q) @ inv(metric) @ jacob_constr(q).T

                where `jacob_constr` is the Jacobian of the constraint function
                `constr` and `metric` is the matrix representation of the
                metric on the ambient space.
            constr (Callable[[array], array]): Function which given a position
                array return as a 1D array the value of the (vector-valued)
                constraint function, the zero level-set of which implicitly
                defines the manifold the dynamic is simulated on.
            metric (None or array or PositiveDefiniteMatrix): Matrix object
                corresponding to matrix representation of metric on
                *unconstrained* position space and covariance of Gaussian
                marginal distribution on *unconstrained* momentum vector. If
                `None` is passed (the default), the identity matrix will be
                used. If a 1D array is passed then this is assumed to specify a
                metric with positive diagonal matrix representation and the
                array the matrix diagonal. If a 2D array is passed then this is
                assumed to specify a metric with a dense positive definite
                matrix representation specified by the array. Otherwise if the
                value is a `mici.matrices.PositiveDefiniteMatrix` subclass it
                is assumed to directly specify the metric matrix
                representation.
            dens_wrt_hausdorff (bool): Whether the `neg_log_dens` function
                specifies the (negative logarithm) of the density of the target
                distribution with respect to the Hausdorff measure on the
                manifold directly (True) or alternatively the negative
                logarithm of a density of a prior distriubtion on the
                unconstrained (ambient) position space with respect to the
                Lebesgue measure, with the target distribution then
                corresponding to the posterior distribution when conditioning
                on the event `const(pos) == 0` (False). Note that in the former
                case the base Hausdorff measure on the manifold depends on the
                metric defined on the ambient space, with the Hausdorff measure
                being defined with respect to the metric induced on the
                manifold from this ambient metric.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct a function to compute the derivative (and value)
                of `neg_log_dens` automatically.
            jacob_constr (
                    None or Callable[[array], array or Tuple[array, array]]):
                Function which given a position array computes the Jacobian
                (matrix / 2D array of partial derivatives) of the output of the
                constraint function `c = constr(q)` with respect to the
                position array argument `q`, returning the computed Jacobian as
                a 2D array `jacob` with

                    jacob[i, j] = ∂c[i] / ∂q[j]

                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the Jacobian
                and the second being the value of `constr` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct a function to compute the Jacobian (and value) of
                `neg_log_dens` automatically.
            mhp_constr (None or
                    Callable[[array], Callable[[array], array]] or
                    Callable[[array], Tuple[Callable, array, array]]):
                Function which given a position array returns another function
                which takes a 2D array as an argument and returns the
                *matrix-Hessian-product* (MHP) of the constraint function
                `constr` with respect to the position array argument. The MHP
                is here defined as a function of a `(dim_constr, dim_pos)`
                shaped 2D array `m`

                    mhp(m) = sum(m[:, :, None] * hess[:, :, :], axis=(0, 1))

                where `hess` is the `(dim_constr, dim_pos, dim_pos)` shaped
                vector-Hessian of `c = constr(q)` with respect to `q` i.e. the
                array of second-order partial derivatives of such that

                    hess[i, j, k] = ∂²c[i] / (∂q[j] ∂q[k])

                Optionally the function may instead return a 3-tuple of values
                with the first a function to compute a MHP of `constr`, the
                second a 2D array corresponding to the Jacobian of `constr`,
                and the third the value of `constr`, all evaluated at the
                passed position array. If `None` is passed (the default) an
                automatic differentiation fallback will be used to attempt to
                construct a function which calculates the MHP (and Jacobian and
                value) of `constr` automatically.
        """
        super().__init__(
            neg_log_dens=neg_log_dens,
            constr=constr,
            metric=metric,
            dens_wrt_hausdorff=dens_wrt_hausdorff,
            grad_neg_log_dens=grad_neg_log_dens,
            jacob_constr=jacob_constr,
        )
        if not dens_wrt_hausdorff:
            self._mhp_constr = autodiff_fallback(
                mhp_constr, constr, "mhp_jacobian_and_value", "mhp_constr"
            )

    @cache_in_state_with_aux("pos", ("jacob_constr", "constr"))
    def mhp_constr(self, state):
        return self._mhp_constr(state.pos)

    def jacob_constr_inner_product(
        self, jacob_constr_1, inner_product_matrix, jacob_constr_2=None
    ):
        if jacob_constr_2 is None or jacob_constr_2 is jacob_constr_1:
            return matrices.DensePositiveDefiniteMatrix(
                jacob_constr_1 @ (inner_product_matrix @ jacob_constr_1.T)
            )
        else:
            return matrices.DenseSquareMatrix(
                jacob_constr_1 @ (inner_product_matrix @ jacob_constr_2.T)
            )

    @cache_in_state("pos")
    def grad_log_det_sqrt_gram(self, state):
        # Evaluate MHP of constraint function before Jacobian as Jacobian value
        # will potentially be computed in 'forward' pass and cached
        mhp_constr = self.mhp_constr(state)
        return mhp_constr(
            self.inv_gram(state) @ self.jacob_constr(state) @ self.metric.inv
        )

####### INTEGRATOR

def maximum_norm(vct):
    """Calculate the maximum (L-infinity) norm of a vector."""
    return np.max(abs(vct))

def solve_projection_onto_manifold_quasi_newton(
    state,
    state_prev,
    dt,
    system,
    constraint_tol=1e-9,
    position_tol=1e-8,
    divergence_tol=1e10,
    max_iters=50,
    norm=maximum_norm,
):
    """Solve constraint equation using quasi-Newton method.

    Uses a quasi-Newton iteration to solve the non-linear system of equations
    in `λ`

        system.constr(
            state.pos + dh2_flow_pos_dmom @
                system.jacob_constr(state_prev).T @ λ) == 0

    where `dh2_flow_pos_dmom = system.dh2_flow_dmom(dt)[0]` is the derivative
    of the action of the (linear) `system.h2_flow` map on the state momentum
    component with respect to the position component, `state` is a post
    (unconstrained) `system.h2_flow` update state with position component
    outside of the manifold and `state_prev` is the corresponding pre-update
    state in the co-tangent bundle.

    Only requires re-evaluating the constraint function `system.constr` within
    the solver loop and no recomputation of matrix decompositions on each
    iteration.

    Args:
        state (mici.states.ChainState): Post `h2_flow `update state to project.
        state_prev (mici.states.ChainState): Previous state in co-tangent
            bundle manifold before `h2_flow` update which defines the
            co-tangent space to perform projection in.
        dt (float): Integrator time step used in `h2_flow` update.
        system (mici.systems.ConstrainedEuclideanMetricSystem): Hamiltonian
           system defining `h2_flow` and `constr` functions used to define
           constraint equation to solve.
        constraint_tol (float): Convergence tolerance in constraint space.
           Iteration will continue until `norm(constr(pos)) < constraint_tol`
           where `pos` is the position at the current iteration.
        position_tol (float): Convergence tolerance in position space.
           Iteration will continue until `norm(delt_pos) < position_tol`
           where `delta_pos` is the change in the position in the current
           iteration.
        divergence_tol (float): Divergence tolerance - solver aborts if
            `norm(constr(pos)) > divergence_tol` on any iteration where `pos`
            is the position at the current iteration and raises
            `mici.errors.ConvergenceError`.
        max_iters (int): Maximum number of iterations to perform before
            aborting and raising `mici.errors.ConvergenceError`.
        norm (Callable[[array], float]): Norm to use to test for convergence.

    Returns:
        Updated `state` object with position component satisfying constraint
        equation to within `constraint_tol`, i.e.
        `norm(system.constr(state.pos)) < constraint_tol`.

    Raises:
        `mici.errors.ConvergenceError` if solver does not converge within
        `max_iters` iterations, diverges or encounters a `ValueError` during
        the iteration.
    """
    mu = np.zeros_like(state.pos)
    jacob_constr_prev = system.jacob_constr(state_prev)
    # Use absolute value of dt and adjust for sign of dt in mom update below
    dh2_flow_pos_dmom, dh2_flow_mom_dmom = system.dh2_flow_dmom(abs(dt))
    inv_jacob_constr_inner_product = system.jacob_constr_inner_product(
        jacob_constr_prev, dh2_flow_pos_dmom
    ).inv
    for i in range(max_iters):
        try:
            constr = system.constr(state)
            state.constr_eval.append(i + 1)
            error = norm(constr)
            delta_mu = jacob_constr_prev.T @ (inv_jacob_constr_inner_product @ constr)
            delta_pos = dh2_flow_pos_dmom @ delta_mu
            if error > divergence_tol or np.isnan(error):
                raise ConvergenceError(
                    f"Quasi-Newton solver diverged on iteration {i}. "
                    f"Last |constr|={error:.1e}, "
                    f"|delta_pos|={norm(delta_pos):.1e}."
                )
            elif error < constraint_tol and norm(delta_pos) < position_tol:
                state.mom -= np.sign(dt) * dh2_flow_mom_dmom @ mu
                return state
            mu += delta_mu
            state.pos -= delta_pos
        except (ValueError, LinAlgError) as e:
            # Make robust to errors in intermediate linear algebra ops
            raise ConvergenceError(
                f"{type(e)} at iteration {i} of quasi-Newton solver ({e})."
            )
    raise ConvergenceError(
        f"Quasi-Newton solver did not converge with {max_iters} iterations. "
        f"Last |constr|={error:.1e}, |delta_pos|={norm(delta_pos)}."
    )


class Integrator(ABC):
    """Base class for integrators."""

    def __init__(self, system, step_size=None):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to integrate the
                dynamics of.
            step_size (float or None): Integrator time step. If set to `None`
                (the default) it is assumed that a step size adapter will be
                used to set the step size before calling the `step` method.
        """
        self.system = system
        self.step_size = step_size

    def step(self, state):
        """Perform a single integrator step from a supplied state.

        Args:
            state (mici.states.ChainState): System state to perform integrator
                step from.

        Returns:
            new_state (mici.states.ChainState): New object corresponding to
                stepped state.
        """
        if self.step_size is None:
            raise AdaptationError(
                "Integrator `step_size` is `None`. This value should only be "
                "used if a step size adapter is being used to set the step "
                "size."
            )
        state = state.copy()
        self._step(state, state.dir * self.step_size)
        return state

    @abstractmethod
    def _step(self, state, dt):
        """Implementation of single integrator step.

        Args:
            state (mici.states.ChainState): System state to perform integrator
                step from. Updated in place.
            dt (float): Integrator time step. May be positive or negative.
        """


class ConstrainedLeapfrogIntegrator(Integrator):
    r"""
    Leapfrog integrator for constrained Hamiltonian systems.

    The Hamiltonian function is assumed to be expressible as the sum of two
    components for which the corresponding (unconstrained) Hamiltonian flows
    can be exactly simulated. Specifically it is assumed that the Hamiltonian
    function \(h\) takes the form

    \[ h(q, p) = h_1(q) + h_2(q, p) \]

    where \(q\) and \(p\) are the position and momentum variables respectively,
    and \(h_1\) and \(h_2\) Hamiltonian component functions for which the exact
    flows can be computed.

    The system is assumed to be additionally subject to a set of holonomic
    constraints on the position component of the state i.e. that all valid
    states must satisfy

    \[ c(q) = 0. \]

    for some differentiable and surjective vector constraint function \(c\) and
    the set of positions satisfying the constraints implicitly defining a
    manifold. There is also a corresponding constraint implied on the momentum
    variables which can be derived by differentiating the above with respect to
    time and using that under the Hamiltonian dynamics the time derivative of
    the position is equal to the negative derivative of the Hamiltonian
    function with respect to the momentum

    \[ \partial c(q) \nabla_2 h(q, p) = 0. \]

    The set of momentum variables satisfying the above for given position
    variables is termed the cotangent space of the manifold (at a position),
    and the set of (position, momentum) pairs for which the position is on the
    constraint manifold and the momentum in the corresponding cotangent space
    is termed the cotangent bundle.

    The integrator exactly preserves these constraints at all steps, such that
    if an initial position momentum pair \((q, p)\) are in the cotangent
    bundle, the corresponding pair after calling the `step` method of the
    integrator will also be in the cotangent bundle.
    """

    def __init__(
        self,
        system,
        step_size=None,
        n_inner_step=1,
        reverse_check_tol=2e-8,
        reverse_check_norm=maximum_norm,
        projection_solver=solve_projection_onto_manifold_quasi_newton,
        projection_solver_kwargs=None,
    ):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to integrate the
                dynamics of.
            step_size (float or None): Integrator time step. If set to `None`
                (the default) it is assumed that a step size adapter will be
                used to set the step size before calling the `step` method.
            n_inner_step (int): Positive integer specifying number of 'inner'
                constrained `system.h2_flow` steps to take within each overall
                step. As the derivative `system.dh1_dpos` is not evaluated
                during the `system.h2_flow` steps, if this derivative is
                relatively expensive to compute compared to evaluating
                `system.h2_flow` then compared to using `n_inner_step = 1` (the
                default) for a given `step_size` it can be more computationally
                efficient to use `n_inner_step > 1` in combination within a
                larger `step_size`, thus reducing the number of
                `system.dh1_dpos` evaluations to simulate forward a given time
                while still controlling the effective time step used for the
                constrained `system.h2_flow` steps which involve solving a
                non-linear system of equations to retract the position
                component of the updated state back on to the manifold, with
                the iterative solver typically diverging if the time step used
                is too large.
            reverse_check_tol (float): Tolerance for check of reversibility of
                implicit sub-steps which involve iterative solving of a
                non-linear system of equations. The step is assumed to be
                reversible if sequentially applying the forward and adjoint
                updates to a state returns to a state with a position component
                within a distance (defined by the `reverse_check_norm`
                argument) of `reverse_check_tol` of the original state position
                component. If this condition is not met a
                `mici.errors.NonReversibleStepError` exception is raised.
            reverse_check_norm (Callable[[array], float]): Norm function
                accepting a single one-dimensional array input and returning a
                non-negative floating point value defining the distance to use
                in the reversibility check. Defaults to
                `mici.solvers.maximum_norm`.
            projection_solver (Callable[
                    [ChainState, ChainState, float, System], ChainState]):
                Function which given two states `state` and `state_prev`,
                floating point time step `dt` and a Hamiltonian system object
                `system` solves the non-linear system of equations in `λ`

                    system.constr(
                        state.pos + dh2_flow_pos_dmom @
                            system.jacob_constr(state_prev).T @ λ) == 0

                where `dh2_flow_pos_dmom = system.dh2_flow_dmom(dt)[0]` is the
                derivative of the action of the (linear) `system.h2_flow` map
                on the state momentum component with respect to the position
                component. This is used to project the state position
                component back on to the manifold after an unconstrained
                `system.h2_flow` update. Defaults to
                `mici.solvers.solve_projection_onto_manifold_quasi_newton`.
            projection_solver_kwargs (None or Dict[str, object]): Dictionary of
                any keyword arguments to `projection_solver`.
        """
        super().__init__(system, step_size)
        self.n_inner_step = n_inner_step
        self.reverse_check_tol = reverse_check_tol
        self.reverse_check_norm = reverse_check_norm
        self.projection_solver = projection_solver
        if projection_solver_kwargs is None:
            projection_solver_kwargs = {}
        self.projection_solver_kwargs = projection_solver_kwargs

    def _h2_flow_retraction_onto_manifold(self, state, state_prev, dt):
        self.system.h2_flow(state, dt)
        self.projection_solver(
            state, state_prev, dt, self.system, **self.projection_solver_kwargs
        )

    def _project_onto_cotangent_space(self, state):
        state.mom = self.system.project_onto_cotangent_space(state.mom, state)

    def _step_a(self, state, dt):
        self.system.h1_flow(state, dt)
        self._project_onto_cotangent_space(state)

    def _step_b(self, state, dt):
        dt_i = dt / self.n_inner_step
        for i in range(self.n_inner_step):
            state_prev = state.copy()
            self._h2_flow_retraction_onto_manifold(state, state_prev, dt_i)
            if i == self.n_inner_step - 1:
                # If at last inner step pre-evaluate dh1_dpos before projecting
                # state on to cotangent space, with computed value being
                # cached. During projection the constraint Jacobian at new
                # position will be calculated however if we are going to make a
                # h1_flow step immediately after we will evaluate dh1_dpos
                # which may involve evaluating the gradient of the log
                # determinant of the Gram matrix, during which we will evaluate
                # the constraint Jacobian in the forward pass anyway.
                # Pre-evaluating here therefore saves one extra Jacobian
                # evaluation when the target density includes a Gram matrix log
                # determinant term (and will not add any cost if this is not
                # the case as dh1_dpos will still be cached and reused).
                self.system.dh1_dpos(state)
            self._project_onto_cotangent_space(state)
            state_back = state.copy()
            self._h2_flow_retraction_onto_manifold(state_back, state, -dt_i)
            rev_diff = self.reverse_check_norm(state_back.pos - state_prev.pos)
            if rev_diff > self.reverse_check_tol:
                raise NonReversibleStepError(
                    f"Non-reversible step. Distance between initial and "
                    f"forward-backward integrated positions = {rev_diff:.1e}."
                )

    def _step(self, state, dt):
        self._step_a(state, 0.5 * dt)
        self._step_b(state, dt)
        self._step_a(state, 0.5 * dt)

#### ADAPTER
class Adapter(ABC):
    """Abstract adapter for implementing schemes to adapt transition parameters.

    Adaptation schemes are assumed to be based on updating a collection of adaptation
    variables (collectively termed the adapter state here) after each chain transition
    based on the sampled chain state and/or statistics of the transition such as an
    acceptance probability statistic. After completing a chain of one or more adaptive
    transitions, the final adapter state may be used to perform a final update to the
    transition parameters.
    """

    @abstractmethod
    def initialize(self, chain_state, transition):
        """Initialize adapter state prior to starting adaptive transitions.

        Args:
            chain_state (mici.states.ChainState): Initial chain state adaptive
                transition will be started from. May be used to calculate initial
                adapter state but should not be mutated by method.
            transition (mici.transitions.Transition): Markov transition being adapted.
                Attributes of the transition or child objects may be updated in-place by
                the method.

        Returns:
            adapt_state (Dict[str, Any]): Initial adapter state.
        """

    @abstractmethod
    def update(self, adapt_state, chain_state, trans_stats, transition):
        """Update adapter state after sampling transition being adapted.

        Args:
            adapt_state (Dict[str, Any]): Current adapter state. Entries will be updated
                in-place by the method.
            chain_state (mici.states.ChainState): Current chain state following sampling
                from transition being adapted. May be used to calculate adapter state
                updates but should not be mutated by method.
            trans_stats (Dict[str, numeric]): Dictionary of statistics associated with
                transition being adapted. May be used to calculate adapter state updates
                but should not be mutated by method.
            transition (mici.transitions.Transition): Markov transition being adapted.
                Attributes of the transition or child objects may be updated in-place by
                the method.
        """

    @abstractmethod
    def finalize(self, adapt_states, chain_states, transition, rngs):
        """Update transition parameters based on final adapter state or states.

        Optionally, if multiple adapter states are available, e.g. from a set of
        independent adaptive chains, then these adaptation information from all the
        chains may be combined to set the transition parameter(s).

        Args:
            adapt_states (Dict[str, Any] or List[Dict[str, Any]]): Final adapter state
                or a list of per chain adapter states. Arrays / buffers associated with
                the adapter state entries may be recycled to reduce memory usage - if so
                the corresponding entries will be removed from the adapter state
                dictionary / dictionaries.
            chain_states (ChainState or List[mici.states.ChainState]): Final state of
                chain (or states of chains) in current sampling stage. May be updated
                in-place if transition parameters altered by adapter require updating
                any state components.
            transition (mici.transitions.Transition): Markov transition being dapted.
                Attributes of the transition or child objects will be updated in-place
                by the method.
            rngs (numpy.random.Generator or List[numpy.random.Generator]): Random number
                generator for the chain or a list of per-chain random number generators.
                Used to resample any components of states needing to be updated due to
                adaptation if required.
        """

    @property
    @abstractmethod
    def is_fast(self):
        """Whether the adapter is 'fast' or 'slow'.

        An adapter which requires only local information to adapt the transition
        parameters should be classified as fast while one which requires more
        global information and so more chain iterations should be classified
        as slow i.e. `is_fast == False`.
        """

class DualAveragingStepSizeAdapter(Adapter):
    """Dual averaging integrator step size adapter.

    Implementation of the dual algorithm step size adaptation algorithm described in
    [1], a modified version of the stochastic optimisation scheme of [2]. By default the
    adaptation is performed to control the `accept_prob` statistic of an integration
    transition to be close to a target value but the statistic adapted on can be altered
    by changing the `adapt_stat_func`.


    References:

      1. Hoffman, M.D. and Gelman, A., 2014. The No-U-turn sampler: adaptively setting
         path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research,
         15(1), pp.1593-1623.
      2. Nesterov, Y., 2009. Primal-dual subgradient methods for convex problems.
         Mathematical programming 120(1), pp.221-259.
    """

    is_fast = True

    def __init__(
        self,
        adapt_stat_target=0.8,
        adapt_stat_func=None,
        log_step_size_reg_target=None,
        log_step_size_reg_coefficient=0.05,
        iter_decay_coeff=0.75,
        iter_offset=10,
        max_init_step_size_iters=100,
    ):
        """
        Args:
            adapt_stat_target (float): Target value for the transition statistic
                being controlled during adaptation.
            adapt_stat_func (Callable[[Dict[str, numeric]], numeric]): Function which
                given a dictionary of transition statistics outputs the value of the
                statistic to control during adaptation. By default this is set to a
                function which simply selects the 'accept_stat' value in the statistics
                dictionary.
            log_step_size_reg_target (float or None): Value to regularize the controlled
                output (logarithm of the integrator step size) towards. If `None` set to
                `log(10 * init_step_size)` where `init_step_size` is the initial
                'reasonable' step size found by a coarse search as recommended in
                Hoffman and Gelman (2014). This has the effect of giving the dual
                averaging algorithm a tendency towards testing step sizes larger than
                the initial value, with typically integrating with a larger step size
                having a lower computational cost.
            log_step_size_reg_coefficient (float): Coefficient controlling amount of
                regularisation of controlled output (logarithm of the integrator step
                size) towards `log_step_size_reg_target`. Defaults to 0.05 as
                recommended in Hoffman and Gelman (2014).
            iter_decay_coeff (float): Coefficient controlling exponent of decay in
                schedule weighting stochastic updates to smoothed log step size
                estimate. Should be in the interval (0.5, 1] to ensure asymptotic
                convergence of adaptation. A value of 1 gives equal weight to the whole
                history of updates while setting to a smaller value increasingly highly
                weights recent updates, giving a tendency to 'forget' early updates.
                Defaults to 0.75 as recommended in Hoffman and Gelman (2014).
            iter_offset (int): Offset used for the iteration based weighting of the
                adaptation statistic error estimate. Should be set to a non-negative
                value. A value > 0 has the effect of stabilising early iterations.
                Defaults to the value of 10 as recommended in Hoffman and Gelman (2014).
            max_init_step_size_iters (int): Maximum number of iterations to use in
                initial search for a reasonable step size with an `AdaptationError`
                exception raised if a suitable step size is not found within this many
                iterations.
        """
        self.adapt_stat_target = adapt_stat_target
        if adapt_stat_func is None:

            def adapt_stat_func(stats):
                return stats["accept_stat"]

        self.adapt_stat_func = adapt_stat_func
        self.log_step_size_reg_target = log_step_size_reg_target
        self.log_step_size_reg_coefficient = log_step_size_reg_coefficient
        self.iter_decay_coeff = iter_decay_coeff
        self.iter_offset = iter_offset
        self.max_init_step_size_iters = max_init_step_size_iters

    def initialize(self, chain_state, transition):
        integrator = transition.integrator
        system = transition.system
        adapt_state = {
            "iter": 0,
            "smoothed_log_step_size": 0.0,
            "adapt_stat_error": 0.0,
        }
        init_step_size = self._find_and_set_init_step_size(
            chain_state, system, integrator
        )
        if self.log_step_size_reg_target is None:
            adapt_state["log_step_size_reg_target"] = log(10 * init_step_size)
        else:
            adapt_state["log_step_size_reg_target"] = self.log_step_size_reg_target
        return adapt_state

    def _find_and_set_init_step_size(self, state, system, integrator):
        """Find initial step size by coarse search using single step statistics.

        Adaptation of Algorithm 4 in Hoffman and Gelman (2014).

        Compared to the Hoffman and Gelman algorithm, this version makes two changes:

          1. The absolute value of the change in Hamiltonian over a step being larger or
             smaller than log(2) is used to determine whether the step size is too big
             or small as opposed to the value of the equivalent Metropolis accept
             probability being larger or smaller than 0.5. Although a negative change in
             the Hamiltonian over a step of magnitude more than log(2) will lead to an
             accept probability of 1 for the forward move, the corresponding reversed
             move will have an accept probability less than 0.5, and so a change in the
             Hamiltonian over a step of magnitude more than log(2) irrespective of the
             sign of the change is indicative of the minimum acceptance probability over
             both forward and reversed steps being less than 0.5.
          2. To allow for integrators for which an integrator step may fail due to e.g.
             a convergence error in an iterative solver, the step size is also
             considered to be too big if any of the step sizes tried in the search
             result in a failed integrator step, with in this case the step size always
             being decreased on subsequent steps irrespective of the initial Hamiltonian
             error, until a integrator step successfully completes and the absolute
             value of the change in Hamiltonian is below the threshold of log(2)
             (corresponding to a minimum acceptance probability over forward and
             reversed steps of 0.5).
        """
        init_state = state.copy()
        h_init = system.h(init_state)
        if np.isnan(h_init):
            raise AdaptationError("Hamiltonian evaluating to NaN at initial state.")
        integrator.step_size = 1
        delta_h_threshold = log(2)
        for s in range(self.max_init_step_size_iters):
            try:
                state = integrator.step(init_state)
                delta_h = abs(h_init - system.h(state))
                if s == 0 or np.isnan(delta_h):
                    step_size_too_big = np.isnan(delta_h) or delta_h > delta_h_threshold
                if (step_size_too_big and delta_h <= delta_h_threshold) or (
                    not step_size_too_big and delta_h > delta_h_threshold
                ):
                    return integrator.step_size
                elif step_size_too_big:
                    integrator.step_size /= 2
                else:
                    integrator.step_size *= 2
            except IntegratorError:
                step_size_too_big = True
                integrator.step_size /= 2
        raise AdaptationError(
            f"Could not find reasonable initial step size in "
            f"{self.max_init_step_size_iters} iterations (final step size "
            f"{integrator.step_size}). A very large final step size may "
            f"indicate that the target distribution is improper such that the "
            f"negative log density is flat in one or more directions while a "
            f"very small final step size may indicate that the density function"
            f" is insufficiently smooth at the point initialized at."
        )

    def update(self, adapt_state, chain_state, trans_stats, transition):
        adapt_state["iter"] += 1
        error_weight = 1 / (self.iter_offset + adapt_state["iter"])
        adapt_state["adapt_stat_error"] *= 1 - error_weight
        adapt_state["adapt_stat_error"] += error_weight * (
            self.adapt_stat_target - self.adapt_stat_func(trans_stats)
        )
        smoothing_weight = (1 / adapt_state["iter"]) ** self.iter_decay_coeff
        log_step_size = adapt_state["log_step_size_reg_target"] - (
            adapt_state["adapt_stat_error"]
            * adapt_state["iter"] ** 0.5
            / self.log_step_size_reg_coefficient
        )
        adapt_state["smoothed_log_step_size"] *= 1 - smoothing_weight
        adapt_state["smoothed_log_step_size"] += smoothing_weight * log_step_size
        transition.integrator.step_size = exp(log_step_size)

    def finalize(self, adapt_states, chain_states, transition, rngs):
        if isinstance(adapt_states, dict):
            transition.integrator.step_size = exp(
                adapt_states["smoothed_log_step_size"]
            )
        else:
            transition.integrator.step_size = sum(
                exp(adapt_state["smoothed_log_step_size"])
                for adapt_state in adapt_states
            ) / len(adapt_states)




#### SAMPLER

def _construct_chain_iterators(
    n_iter, chain_iterator_class, n_chain=None, position_offset=0
):
    """Set up chain iterator progress bar object(s)."""
    if n_chain is None:
        return chain_iterator_class(range(n_iter), description="Chain 1/1")
    else:
        return [
            chain_iterator_class(
                range(n_iter),
                description=f"Chain {c+1}/{n_chain}",
                position=(c + position_offset, n_chain + position_offset),
            )
            for c in range(n_chain)
        ]

def _check_and_process_init_state(state, transitions):
    """Check initial chain state is valid and convert dict to ChainState."""
    for trans_key, transition in transitions.items():
        for var_key in transition.state_variables:
            if var_key not in state:
                raise ValueError(
                    f"init_state does contain have {var_key} value required by"
                    f" {trans_key} transition."
                )
    if not isinstance(state, (ChainState, dict)):
        raise TypeError("init_state should be a dictionary or ChainState.")
    return ChainState(**state) if isinstance(state, dict) else state

def _get_valid_filename(string):
    """Generate a valid filename from a string.

    Strips all characters which are not alphanumeric or a period (.), dash (-)
    or underscore (_).

    Based on https://stackoverflow.com/a/295146/4798943

    Args:
        string (str): String file name to process.

    Returns:
        str: Generated file name.
    """
    return "".join(c for c in string if (c.isalnum() or c in "._- "))

def _generate_memmap_filename(dir_path, prefix, key, index):
    """Generate a new memory-map filename."""
    key_str = _get_valid_filename(str(key))
    return os.path.join(dir_path, f"{prefix}_{index}_{key_str}.npy")

def _open_new_memmap(file_path, shape, default_val, dtype):
    """Open a new memory-mapped array object and fill with a default-value.

    Args:
        file_path (str): Path to write memory-mapped array to.
        shape (Tuple[int, ...]): Shape of new array.
        default_val: Value to fill array with. Should be compatible with
            specified `dtype`.
        dtype (str or numpy.dtype): NumPy data-type for array.

    Returns
        memmap (numpy.memmap): Memory-mapped array object.
    """
    if isinstance(shape, int):
        shape = (shape,)
    memmap = np.lib.format.open_memmap(file_path, dtype=dtype, mode="w+", shape=shape)
    memmap[:] = default_val
    return memmap

def _init_chain_stats(transitions, n_iter, memmap_enabled, memmap_path, chain_index):
    """Initialize dictionary of per-transition chain statistics array dicts."""
    chain_stats = {}
    for trans_key, transition in transitions.items():
        chain_stats[trans_key] = {}
        if transition.statistic_types is not None:
            for key, (dtype, val) in transition.statistic_types.items():
                if memmap_enabled:
                    filename = _generate_memmap_filename(
                        memmap_path, "stats", f"{trans_key}_{key}", chain_index
                    )
                    chain_stats[trans_key][key] = _open_new_memmap(
                        filename, n_iter, val, dtype
                    )
                else:
                    chain_stats[trans_key][key] = np.full(n_iter, val, dtype)
    return chain_stats

def _init_traces(
    trace_funcs, init_state, n_iter, memmap_enabled, memmap_path, chain_index
):
    """Initialize dictionary of chain trace arrays."""
    traces = {}
    for trace_func in trace_funcs:
        for key, val in trace_func(init_state).items():
            val = np.array(val) if np.isscalar(val) else val
            init = np.nan if np.issubdtype(val.dtype, np.inexact) else 0
            if memmap_enabled:
                filename = _generate_memmap_filename(
                    memmap_path, "trace", key, chain_index
                )
                traces[key] = _open_new_memmap(
                    filename, (n_iter,) + val.shape, init, val.dtype
                )
            else:
                traces[key] = np.full((n_iter,) + val.shape, init, val.dtype)
    return traces

def _get_obj_byte_size(obj, seen=None):
    """Recursively finds size of objects in bytes.

    Original source: https://github.com/bosswissam/pysize

    MIT License

    Copyright (c) [2018] [Wissam Jarjoui]

    Args:
        obj (object): Object to calculate size of.
        seen (None or Set): Set of objects seen so far.

    Returns:
        int: Byte size of `obj`.
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, "__dict__"):
        for cls in obj.__class__.__mro__:
            if "__dict__" in cls.__dict__:
                d = cls.__dict__["__dict__"]
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += _get_obj_byte_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((_get_obj_byte_size(v, seen) for v in obj.values()))
        size += sum((_get_obj_byte_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum((_get_obj_byte_size(i, seen) for i in obj))

    if hasattr(obj, "__slots__"):  # can have __slots__ with __dict__
        size += sum(
            _get_obj_byte_size(getattr(obj, s), seen)
            for s in obj.__slots__
            if hasattr(obj, s)
        )

    return size

def _check_chain_data_size(traces, chain_stats):
    """Check that byte size of sample chain calls below pickle limit."""
    total_return_nbytes = _get_obj_byte_size(traces) + _get_obj_byte_size(chain_stats)
    # Check if total number of bytes to be returned exceeds pickle limit
    if total_return_nbytes > 2 ** 31 - 1:
        raise RuntimeError(
            f"Total number of bytes allocated for chain data to be returned "
            f"({total_return_nbytes / 2**30:.2f} GiB) exceeds size limit for "
            f"returning results of a process (2 GiB). Try rerunning with "
            f" memory-mapping enabled (`memmap_enabled=True`)."
        )

def _update_chain_stats(sample_index, chain_stats, trans_key, trans_stats):
    """Update chain statistics arrays for current chain iteration."""
    if trans_stats is not None:
        if sample_index == 0 and trans_key not in chain_stats:
            raise KeyError(
                f"Transition {trans_key} returned statistics but has no "
                f"statistic_types attribute."
            )
        for key, val in trans_stats.items():
            if sample_index == 0 and key not in chain_stats[trans_key]:
                raise KeyError(
                    f"Transition {trans_key} returned {key} statistic but it "
                    f"is not included in its statistic_types attribute."
                )
            chain_stats[trans_key][key][sample_index] = val

def _update_monitor_stats(sample_index, chain_stats, monitor_stats, monitor_dict):
    """Update dictionary of per-iteration monitored statistics."""
    for (trans_key, stats_key) in monitor_stats:
        if sample_index == 0 and not (
            trans_key in chain_stats and stats_key in chain_stats[trans_key]
        ):
            raise KeyError(
                f"Statistics key pair {(trans_key, stats_key)} to be monitored"
                "is not present in chain statistics returned by transitions."
            )
        val = chain_stats[trans_key][stats_key][sample_index]
        if stats_key not in monitor_dict:
            monitor_dict[stats_key] = val
        else:
            monitor_dict[f"{trans_key}.{stats_key}"] = val

def _try_resize_dim_0_inplace(array, new_shape_0):
    """Try to resize 0-axis of array in-place or return a view otherwise."""
    if new_shape_0 >= array.shape[0]:
        return array
    try:
        # Try to truncate arrays by resizing in place
        array.resize((new_shape_0,) + array.shape[1:])
        return array
    except ValueError:
        # In place resize not possible therefore return truncated view
        return array[:new_shape_0]

def _truncate_chain_data(sample_index, traces, chain_stats):
    """Truncate first dimension of chain arrays to sample_index < n_iter."""
    for key in traces:
        traces[key] = _try_resize_dim_0_inplace(traces[key], sample_index)
    for trans_stats in chain_stats.values():
        for key in trans_stats:
            trans_stats[key] = _try_resize_dim_0_inplace(trans_stats[key], sample_index)

def _flush_memmap_chain_data(traces, chain_stats):
    """Flush all pending writes to memory-mapped chain data arrays to disk."""
    for trace in traces.values():
        trace.flush()
    for trans_stats in chain_stats.values():
        for stat in trans_stats.values():
            stat.flush()

def _memmaps_to_file_paths(obj):
    """Convert memmap objects to corresponding file paths.

    Acts recursively on arbitrary 'pytrees' of nested dict/tuple/lists with
    memmap leaves.

    Arg:
        obj: NumPy memmap object or pytree of memmap objects to convert.

    Returns:
        File path string or pytree of file path strings.
    """
    if isinstance(obj, np.memmap):
        return obj.filename
    elif isinstance(obj, dict):
        return {k: _memmaps_to_file_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_memmaps_to_file_paths(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_memmaps_to_file_paths(v) for v in obj)

def _sample_chain(
    init_state,
    chain_iterator,
    rng,
    transitions,
    trace_funcs,
    chain_index=0,
    parallel_chains=False,
    memmap_enabled=False,
    memmap_path=None,
    monitor_stats=None,
    adapters=None,
):
    """Sample a chain by iteratively appyling a sequence of transition kernels.

    Args:
        init_state (mici.states.ChainState or Dict[str, object]): Initial chain
            state. Either a `mici.states.ChainState` object or a dictionary
            with entries specifying initial values for all state variables used
            by chain transition `sample` methods.
        chain_iterator (Iterable[Tuple[int, Dict]]): Iterable object which
            is iterated over to produce sample indices and (empty) iteration
            statistic dictionaries to output monitored chain statistics to
            during sampling.
        rng (numpy.random.Generator): Numpy random number generator.
        transitions (OrderedDict[str, Transition]): Ordered dictionary of
            Markov transitions kernels to sequentially sample from on each
            chain iteration.
        trace_funcs (Iterable[Callable[[ChainState], Dict[str, array]]]):
            List of functions which compute the variables to be recorded at
            each chain iteration, with each trace function being passed the
            current state and returning a dictionary of scalar or array values
            corresponding to the variable(s) to be stored. The keys in the
            returned dictionaries are used to index the trace arrays in the
            returned traces dictionary. If a key appears in multiple
            dictionaries only the the value corresponding to the last trace
            function to return that key will be stored.
        chain_index (int): Identifier for chain when sampling multiple chains.
        parallel_chains (bool): Whether multiple chains are being sampled in
            parallel.
        memmap_enabled (bool): Whether to memory-map arrays used to store chain
            data to files on disk to avoid excessive system memory usage for
            long chains and/or large chain states. The chain data is written to
            `.npy` files in the directory specified by `memmap_path` (or a
            temporary directory if not provided). These files persist after the
            termination of the function so should be manually deleted when no
            longer required. Default is to for memory mapping to be disabled.
        memmap_path (str): Path to directory to write memory-mapped chain data
            to. If not provided, a temporary directory will be created and the
            chain data written to files there.
        monitor_stats (Iterable[Tuple[str, str]]): List of tuples of string key
            pairs, with first entry the key of a Markov transition in the
            `transitions` dict passed to the the `__init__` method and the
            second entry the key of a chain statistic that will be returned in
            the `chain_stats` dictionary. The mean over samples computed so far
            of the chain statistics associated with any valid key-pairs will be
            monitored during sampling by printing as postfix to progress bar.
        adapters (Dict[str, Iterable[Adapter]): Dictionary of iterables of
            `mici.adapters.Adapter` instances keyed by strings corresponding to
            the key of the transition in the `transitions` dictionary to apply
            the adapters to. Each adapter is able to adaptatively set the
            parameters of a transition while sampling a chain. Note that the
            adapter updates for each transition are applied in the order the
            adapters appear in the iterable and so if multiple adapters change
            the same parameter(s) the order will matter. Adaptation based on the
            chain state history breaks the Markov property and so any chain
            samples while adaptation is active should not be used in estimates
            of expectations.

    Returns:
        final_state (mici.states.ChainState): State of chain after final
            iteration. May be used to resume sampling a chain by passing as the
            initial state to a new `sample_chain` call.
        traces (Dict[str, array]): Dictionary of chain trace arrays. Values in
            dictionary are arrays of variables outputted by trace functions in
            `trace_funcs` with leading dimension of array corresponding to the
            sampling (draw) index. The key for each value is the corresponding
            key in the dictionary returned by the trace function which computed
            the traced value.
        chain_stats (Dict[str, Dict[str, array]]): Dictionary of chain
            transition statistic dictionaries. Values in outer dictionary are
            dictionaries of statistics for each chain transition, keyed by the
            string key for the transition. The values in each inner transition
            dictionary are arrays of chain statistic values with the leading
            dimension of each array corresponding to the sampling (draw) index.
            The key for each value is a string description of the corresponding
            integration transition statistic.
        exception (None or Exception): Any handled exception which may affect
            how the returned outputs are processed by the caller.
    """
    state = _check_and_process_init_state(init_state, transitions)
    n_iter = len(chain_iterator)
    # Create temporary directory if memory mapping and no path provided
    if memmap_enabled and memmap_path is None:
        memmap_path = tempfile.mkdtemp()
    chain_stats = _init_chain_stats(
        transitions, n_iter, memmap_enabled, memmap_path, chain_index
    )
    traces = _init_traces(
        trace_funcs, state, n_iter, memmap_enabled, memmap_path, chain_index
    )
    adapter_states = {}
    try:
        if adapters is not None:
            for trans_key, adapter_list in adapters.items():
                adapter_states[trans_key] = []
                for adapter in adapter_list:
                    adapter_states[trans_key].append(
                        adapter.initialize(state, transitions[trans_key])
                    )
    except AdaptationError as exception:
        logger.error(
            f"Initialisation of {type(adapter).__name__} for chain "
            f"{chain_index + 1} failed: {exception}"
        )
        return state, traces, chain_stats, adapter_states, exception
    try:
        sample_index = 0
        if parallel_chains and not memmap_enabled:
            _check_chain_data_size(traces, chain_stats)
        with chain_iterator:
            for sample_index, monitor_dict in chain_iterator:
                for trans_key, transition in transitions.items():
                    state, trans_stats = transition.sample(state, rng)
                    if adapters is not None and trans_key in adapters:
                        for adapter, adapter_state in zip(
                            adapters[trans_key], adapter_states[trans_key]
                        ):
                            adapter.update(
                                adapter_state, state, trans_stats, transition
                            )
                    _update_chain_stats(
                        sample_index, chain_stats, trans_key, trans_stats
                    )
                for trace_func in trace_funcs:
                    for key, val in trace_func(state).items():
                        traces[key][sample_index] = val
                if monitor_stats is not None:
                    _update_monitor_stats(
                        sample_index, chain_stats, monitor_stats, monitor_dict
                    )
    except KeyboardInterrupt as e:
        exception = e
        logger.error(
            f"Sampling manually interrupted for chain {chain_index + 1} at"
            f" iteration {sample_index}. Arrays containing chain traces and"
            f" statistics computed before interruption will be returned."
        )
        # Sampling interrupted therefore truncate returned arrays unless using
        # memory mapping with parallel chains as will only return file paths
        if not (parallel_chains and memmap_enabled):
            _truncate_chain_data(sample_index, traces, chain_stats)
    else:
        exception = None
    if memmap_enabled:
        _flush_memmap_chain_data(traces, chain_stats)
    if parallel_chains and memmap_enabled:
        traces = _memmaps_to_file_paths(traces)
        chain_stats = _memmaps_to_file_paths(chain_stats)
    return state, traces, chain_stats, adapter_states, exception

def _finalize_adapters(adapter_states_dict, chain_states, adapters, transitions, rngs):
    """Finalize adapter updates to transitions based on final adapter states."""
    for trans_key, adapter_states_list in adapter_states_dict.items():
        for adapter_states, adapter in zip(adapter_states_list, adapters[trans_key]):
            adapter.finalize(adapter_states, chain_states, transitions[trans_key], rngs)

def _get_per_chain_rngs(base_rng, n_chain):
    """Construct random number generators (RNGs) for each of a set of chains.

    If the base RNG bit generator has a `jumped` method this is used to produce
    a sequence of independent random substreams. Otherwise if the base RNG bit
    generator has a `_seed_seq` attribute this is used to spawn a sequence off
    generators.
    """
    if hasattr(base_rng, "bit_generator"):
        bit_generator = base_rng.bit_generator
    elif hasattr(base_rng, "_bit_generator"):
        bit_generator = base_rng._bit_generator
    else:
        bit_generator = None
    if bit_generator is not None and hasattr(bit_generator, "jumped"):
        return [default_rng(bit_generator.jumped(i)) for i in range(n_chain)]
    elif bit_generator is not None and hasattr(bit_generator, "_seed_seq"):
        seed_sequence = bit_generator._seed_seq
        return [default_rng(seed) for seed in seed_sequence.spawn(n_chain)]
    else:
        raise ValueError(f"Unsupported random number generator type {type(base_rng)}.")

def _collate_chain_outputs(chain_outputs):
    """Unzip list of tuples of chain outputs in to tuple of lists of outputs.

    As well as collating chain outputs, any string file path values
    corresponding to memmap file paths are swapped for the corresponding
    memmap objects.
    """
    final_states_stack = []
    traces_stack = {}
    stats_stack = {}
    adapt_states_stack = {}
    for chain_index, (final_state, traces, stats, adapt_states) in enumerate(
        chain_outputs
    ):
        final_states_stack.append(final_state)
        for key, val in traces.items():
            # if value is string => file path to memory mapped array
            if isinstance(val, str):
                val = np.lib.format.open_memmap(val)
            if chain_index == 0:
                traces_stack[key] = [val]
            else:
                traces_stack[key].append(val)
        for trans_key, trans_stats in stats.items():
            if chain_index == 0:
                stats_stack[trans_key] = {}
            for key, val in trans_stats.items():
                # if value is string => file path to memory mapped array
                if isinstance(val, str):
                    val = np.lib.format.open_memmap(val)
                if chain_index == 0:
                    stats_stack[trans_key][key] = [val]
                else:
                    stats_stack[trans_key][key].append(val)
        for trans_key, adapt_state_list in adapt_states.items():
            if trans_key not in adapt_states_stack:
                adapt_states_stack[trans_key] = [[a] for a in adapt_state_list]
            else:
                for i, adapt_state in enumerate(adapt_state_list):
                    adapt_states_stack[trans_key][i].append(adapt_state)
    return final_states_stack, traces_stack, stats_stack, adapt_states_stack

def _sample_chains_sequential(init_states, rngs, chain_iterators, **kwargs):
    """Sample multiple chains sequentially in a single process."""
    chain_outputs = []
    exception = None
    for chain_index, (init_state, rng, chain_iterator) in enumerate(
        zip(init_states, rngs, chain_iterators)
    ):
        *outputs, exception = _sample_chain(
            chain_iterator=chain_iterator,
            init_state=init_state,
            rng=rng,
            chain_index=chain_index,
            parallel_chains=False,
            **kwargs,
        )
        # Returned exception being AdaptationError indicates chain terminated
        # due to adapter initialisation failing therefore do not store returned
        # chain outputs
        if not isinstance(exception, AdaptationError):
            chain_outputs.append(outputs)
        # If returned handled exception was a manual interrupt break and return
        if isinstance(exception, KeyboardInterrupt):
            break
    return (*_collate_chain_outputs(chain_outputs), exception)

def _ignore_sigint_initializer():
    """Initializer for processes to force ignoring SIGINT interrupt signals."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

@contextmanager
def _ignore_sigint_manager():
    """Context-managed SyncManager which ignores SIGINT interrupt signals."""
    manager = SyncManager()
    try:
        manager.start(_ignore_sigint_initializer)
        yield manager
    finally:
        manager.shutdown()

def _sample_chains_worker(chain_queue, iter_queue):
    """Worker process function for parallel sampling of chains.

    Consumes chain arguments from a shared queue and outputs chain progress
    updates to a second shared queue.
    """
    chain_outputs = []
    while not chain_queue.empty():
        try:
            chain_index, init_state, rng, n_iter, kwargs = chain_queue.get(block=False)
            max_threads = kwargs.pop("max_threads_per_process", None)
            context = (
                threadpool_limits(limits=max_threads)
                if THREADPOOLCTL_AVAILABLE
                else nullcontext()
            )
            with context:
                *outputs, exception = _sample_chain(
                    init_state=init_state,
                    rng=rng,
                    chain_index=chain_index,
                    chain_iterator=_ProxyProgressBar(
                        range(n_iter), chain_index, iter_queue
                    ),
                    parallel_chains=True,
                    **kwargs,
                )
            # Returned exception being AdaptationError indicates chain
            # terminated due to adapter initialisation failing therefore do not
            # store returned chain outputs and put None value on iteration queue
            # to indicate to parent process chain terminated
            if isinstance(exception, AdaptationError):
                iter_queue.put(None)
            else:
                chain_outputs.append((chain_index, outputs))
            # If returned handled exception was a manual interrupt put exception
            # on iteration queue to communicate to parent process and break
            if isinstance(exception, KeyboardInterrupt):
                iter_queue.put(exception)
                break
        except queue.Empty:
            pass
        except Exception as exception:
            # Log exception here so that correct traceback is logged
            logger.error(
                "Exception encountered in chain worker process:", exc_info=exception
            )
            # Put exception on iteration queue to be reraised in parent process
            iter_queue.put(exception)
    return chain_outputs

def _sample_chains_parallel(init_states, rngs, chain_iterators, n_process, **kwargs):
    """Sample multiple chains in parallel over multiple processes."""
    n_iters = [len(it) for it in chain_iterators]
    n_chain = len(chain_iterators)
    with _ignore_sigint_manager() as manager, Pool(n_process) as pool:
        results = None
        exception = None
        try:
            # Shared queue for workers to output chain progress updates to
            iter_queue = manager.Queue()
            # Shared queue for workers to get arguments for _sample_chain calls
            # from on initialising each chain
            chain_queue = manager.Queue()
            for c, (init_state, rng, n_iter) in enumerate(
                zip(init_states, rngs, n_iters)
            ):
                chain_queue.put((c, init_state, rng, n_iter, kwargs))
            # Start n_process worker processes which each have access to the
            # shared queues, returning results asynchronously
            results = pool.starmap_async(
                _sample_chains_worker,
                [(chain_queue, iter_queue) for p in range(n_process)],
            )
            # Start loop to use chain progress updates outputted to iter_queue
            # by worker processes to update progress bars, using an ExitStack
            # to ensure all progress bars are context managed so that they
            # are closed properly on for example manual interrupts
            with ExitStack() as stack:
                pbars = [stack.enter_context(it) for it in chain_iterators]
                # Deadlock seems to occur when directly using results.ready()
                # method to check if all chains completed sampling in
                # while loop condition therefore manually keep track of
                # number of completed chains
                chains_completed = 0
                while not (iter_queue.empty() and chains_completed == n_chain):
                    iter_queue_item = iter_queue.get()
                    # Queue item being None indicates a chain terminated early
                    # due to a non-fatal error e.g. an error in initialising an
                    # adapter. In this case we continue sampling any other
                    # remaining chains but increment the completed chains
                    # counter to ensure correct termination of chain progress
                    # update loop
                    if iter_queue_item is None:
                        chains_completed += 1
                    # If queue item is KeyboardInterrupt exception break out of
                    # chain progress update loop but do not reraise exception
                    # so that partial chain outputs are returned
                    elif isinstance(iter_queue_item, KeyboardInterrupt):
                        exception = iter_queue_item
                        break
                    # Re raise any other exception passed from worker processes
                    elif isinstance(iter_queue_item, Exception):
                        raise RuntimeError(
                            "Unhandled exception in chain worker process."
                        ) from iter_queue_item
                    else:
                        # Otherwise unpack and update progress bar
                        chain_index, sample_index, data_dict = iter_queue_item
                        pbars[chain_index].update(sample_index, data_dict)
                        if sample_index == n_iters[chain_index]:
                            chains_completed += 1
        except (PicklingError, AttributeError) as e:
            if not MULTIPROCESS_AVAILABLE and (
                isinstance(e, PicklingError) or "pickle" in str(e)
            ):
                raise RuntimeError(
                    "Error encountered while trying to run chains on multiple"
                    "processes in parallel. The inbuilt multiprocessing module"
                    " uses pickle to communicate between processes and pickle "
                    "does support pickling anonymous or nested functions. If "
                    "you use anonymous or nested functions in your model "
                    "functions or are using autograd to automatically compute "
                    "derivatives (autograd uses anonymous and nested "
                    "functions) then installing the Python package "
                    "multiprocess, which is able to serialise anonymous and "
                    "nested functions and will be used in preference to "
                    "multiprocessing by this package when available, may "
                    "resolve this error."
                ) from e
            else:
                raise e
        except KeyboardInterrupt as e:
            # Interrupts handled in child processes therefore ignore here
            exception = e
        if results is not None:
            # Join all output lists from per-process workers in to single list
            indexed_chain_outputs = sum((res for res in results.get()), [])
            # Sort list by chain index (first element of tuple entries) and
            # then create new list with chain index removed
            chain_outputs = [outp for i, outp in sorted(indexed_chain_outputs)]
        else:
            chain_outputs = []
    return (*_collate_chain_outputs(chain_outputs), exception)



class MarkovChainMonteCarloMethod(object):
    """Generic Markov chain Monte Carlo (MCMC) sampler.

    Generates a Markov chain from some initial state by iteratively applying
    a sequence of Markov transition operators.
    """

    def __init__(self, rng, transitions):
        """
        Args:
            rng (numpy.random.Generator): Numpy random number generator.
            transitions (OrderedDict[str, Transition]): Ordered dictionary of
                Markov transitions kernels to sequentially sample from on each
                chain iteration.
        """
        if isinstance(rng, np.random.RandomState):
            warn(
                "Use of numpy.random.RandomState random number generators is "
                "deprecated. Please use a numpy.random.Generator instance "
                "instead for example from a call to numpy.random.default_rng.",
                DeprecationWarning,
            )
            rng = np.random.Generator(rng._bit_generator)
        self.rng = rng
        self.transitions = transitions

    def __set_sample_chain_kwargs_defaults(self, kwargs):
        if "memmap_enabled" not in kwargs:
            kwargs["memmap_enabled"] = False
        if kwargs["memmap_enabled"] and kwargs.get("memmap_path") is None:
            kwargs["memmap_path"] = tempfile.mkdtemp()
        display_progress = kwargs.pop("display_progress", True)
        if not display_progress:
            kwargs["progress_bar_class"] = DummyProgressBar
        elif "progress_bar_class" not in kwargs:
            kwargs["progress_bar_class"] = ProgressBar

    def sample_chain(self, n_iter, init_state, trace_funcs, **kwargs):
        """Sample a Markov chain from a given initial state.

        Performs a specified number of chain iterations (each of which may be
        composed of multiple individual Markov transitions), recording the
        outputs of functions of the sampled chain state after each iteration.

        Args:
            n_iter (int): Number of iterations (samples to draw) per chain.
            init_state (mici.states.ChainState or Dict[str, object]): Initial
                chain state. Either a `mici.states.ChainState` object or a
                dictionary with entries specifying initial values for all state
                variables used by chain transition `sample` methods.
            trace_funcs (Iterable[Callable[[ChainState], Dict[str, array]]]):
                List of functions which compute the variables to be recorded at
                each chain iteration, with each trace function being passed the
                current state and returning a dictionary of scalar or array
                values corresponding to the variable(s) to be stored. The keys
                in the returned dictionaries are used to index the trace arrays
                in the returned traces dictionary. If a key appears in multiple
                dictionaries only the the value corresponding to the last trace
                function to return that key will be stored.

        Kwargs:
            memmap_enabled (bool): Whether to memory-map arrays used to store
                chain data to files on disk to avoid excessive system memory
                usage for long chains and/or large chain states. The chain data
                is written to `.npy` files in the directory specified by
                `memmap_path` (or a temporary directory if not provided). These
                files persist after the termination of the function so should
                be manually deleted when no longer required. Default is to
                for memory mapping to be disabled.
            memmap_path (str): Path to directory to write memory-mapped chain
                data to. If not provided, a temporary directory will be created
                and the chain data written to files there.
            monitor_stats (Iterable[Tuple[str, str]]): List of tuples of string
                key pairs, with first entry the key of a Markov transition in
                the `transitions` dict passed to the the `__init__` method and
                the second entry the key of a chain statistic that will be
                returned in the `chain_stats` dictionary. The mean over samples
                computed so far of the chain statistics associated with any
                valid key-pairs will be monitored during sampling by printing
                as postfix to progress bar.
            display_progress (bool): Whether to display a progress bar to
                track the completed chain sampling iterations. Default value
                is `True`, i.e. to display progress bar.
            progress_bar_class (mici.progressbars.BaseProgressBar): Class or
                factory function for progress bar to use to show chain
                progress if enabled (`display_progress=True`). Defaults to
                `mici.progressbars.ProgressBar`.
            adapters (Dict[str, Iterable[Adapter]): Dictionary of iterables of
                `mici.adapters.Adapter` instances keyed by strings corresponding
                to the key of the transition in the `transitions` dictionary to
                apply the adapters to. Each adapter is able to adaptatively set
                the parameters of a transition while sampling a chain. Note that
                the adapter updates for each transition are applied in the order
                the adapters appear in the iterable and so if multiple adapters
                change the same parameter(s) the order will matter. Adaptation
                based on the chain state history breaks the Markov property and
                so any chain samples while adaptation is active should not be
                used in estimates of expectations.

        Returns:
            final_state (mici.states.ChainState): State of chain after final
                iteration. May be used to resume sampling a chain by passing as
                the initial state to a new `sample_chain` call.
            traces (Dict[str, array]): Dictionary of chain trace arrays. Values
                in dictionary are arrays of variables outputted by trace
                functions in `trace_funcs` with leading dimension of array
                corresponding to the sampling (draw) index. The key for each
                value is the corresponding key in the dictionary returned by
                the trace function which computed the traced value.
            chain_stats (Dict[str, Dict[str, array]]): Dictionary of chain
                transition statistic dictionaries. Values in outer dictionary
                are dictionaries of statistics for each chain transition, keyed
                by the string key for the transition. The values in each inner
                transition dictionary are arrays of chain statistic values with
                the leading dimension of each array corresponding to the
                sampling (draw) index. The key for each value is a string
                description of the corresponding integration transition
                statistic.
        """
        self.__set_sample_chain_kwargs_defaults(kwargs)
        chain_iterator = _construct_chain_iterators(
            n_iter, kwargs.pop("progress_bar_class")
        )
        final_state, traces, chain_stats, adapter_states, _ = _sample_chain(
            init_state=init_state,
            chain_iterator=chain_iterator,
            transitions=self.transitions,
            rng=self.rng,
            trace_funcs=trace_funcs,
            parallel_chains=False,
            **kwargs,
        )
        if len(adapter_states) > 0:
            _finalize_adapters(
                adapter_states,
                final_state,
                kwargs["adapters"],
                self.transitions,
                self.rng,
            )

        return final_state, traces, chain_stats

    def sample_chains(self, n_iter, init_states, trace_funcs, n_process=1, **kwargs):
        """Sample one or more Markov chains from given initial states.

        Performs a specified number of chain iterations (each of which may be
        composed of multiple individual Markov transitions), recording the
        outputs of functions of the sampled chain state after each iteration.
        The chains may be run in parallel across multiple independent processes
        or sequentially. In all cases all chains use independent random draws.

        Args:
            n_iter (int): Number of iterations (samples to draw) per chain.
            init_states (Iterable[ChainState] or Iterable[Dict[str, object]]):
                Initial chain states. Each entry can be either a `ChainState`
                object or a dictionary with entries specifying initial values
                for all state variables used by chain transition `sample`
                methods.
            trace_funcs (Iterable[Callable[[ChainState], Dict[str, array]]]):
                List of functions which compute the variables to be recorded at
                each chain iteration, with each trace function being passed the
                current state and returning a dictionary of scalar or array
                values corresponding to the variable(s) to be stored. The keys
                in the returned dictionaries are used to index the trace arrays
                in the returned traces dictionary. If a key appears in multiple
                dictionaries only the the value corresponding to the last trace
                function to return that key will be stored.
            n_process (int or None): Number of parallel processes to run chains
                over. If set to one then chains will be run sequentially in
                otherwise a `multiprocessing.Pool` object will be used to
                dynamically assign the chains across multiple processes. If
                set to `None` then the number of processes will default to the
                output of `os.cpu_count()`.

        Kwargs:
            max_threads_per_process (int or None): If `threadpoolctl` is
                available this argument may be used to limit the maximum number
                of threads that can be used in thread pools used in libraries
                supported by `threadpoolctl`, which include BLAS and OpenMP
                implementations. This argument will only have an effect if
                `n_process > 1` such that chains are being run on multiple
                processes and only if `threadpoolctl` is installed in the
                current Python environment. If set to `None` (the default) no
                limits are set.
            memmap_enabled (bool): Whether to memory-map arrays used to store
                chain data to files on disk to avoid excessive system memory
                usage for long chains and/or large chain states. The chain data
                is written to `.npy` files in the directory specified by
                `memmap_path` (or a temporary directory if not provided). These
                files persist after the termination of the function so should
                be manually deleted when no longer required. Default is to
                for memory mapping to be disabled.
            memmap_path (str): Path to directory to write memory-mapped chain
                data to. If not provided, a temporary directory will be created
                and the chain data written to files there.
            monitor_stats (Iterable[Tuple[str, str]]): List of tuples of string
                key pairs, with first entry the key of a Markov transition in
                the `transitions` dict passed to the the `__init__` method and
                the second entry the key of a chain statistic that will be
                returned in the `chain_stats` dictionary. The mean over samples
                computed so far of the chain statistics associated with any
                valid key-pairs will be monitored during sampling  by printing
                as postfix to progress bar.
            display_progress (bool): Whether to display a progress bar to
                track the completed chain sampling iterations. Default value
                is `True`, i.e. to display progress bar.
            progress_bar_class (mici.progressbars.BaseProgressBar): Class or
                factory function for progress bar to use to show chain
                progress if enabled (`display_progress=True`). Defaults to
                `mici.progressbars.ProgressBar`.
            adapters (Dict[str, Iterable[Adapter]): Dictionary of iterables of
                `mici.adapters.Adapter` instances keyed by strings corresponding
                to the key of the transition in the `transitions` dictionary to
                apply the adapters to. Each adapter is able to adaptatively set
                the parameters of a transition while sampling a chain. Note that
                the adapter updates for each transition are applied in the order
                the adapters appear in the iterable and so if multiple adapters
                change the same parameter(s) the order will matter. Adaptation
                based on the chain state history breaks the Markov property and
                so any chain samples while adaptation is active should not be
                used in estimates of expectations.

        Returns:
            final_states (List[ChainState]): States of chains after final
                iteration. May be used to resume sampling a chain by passing as
                the initial states to a new `sample_chains` call.
            traces (Dict[str, List[array]]): Dictionary of chain trace arrays.
                Values in dictionary are list of arrays of variables outputted
                by trace functions in `trace_funcs` with each array in the list
                corresponding to a single chain and the leading dimension of
                each array corresponding to the sampling (draw) index. The key
                for each value is the corresponding key in the dictionary
                returned by the trace function which computed the traced value.
            chain_stats (Dict[str, Dict[str, List[array]]]): Dictionary of
                chain transition statistic dictionaries. Values in outer
                dictionary are dictionaries of statistics for each chain
                transition, keyed by the string key for the transition. The
                values in each inner transition dictionary are lists of arrays
                of chain statistic values with each array in the list
                corresponding to a single chain and the leading dimension of
                each array corresponding to the sampling (draw) index. The key
                for each value is a string description of the corresponding
                integration transition statistic.
        """
        self.__set_sample_chain_kwargs_defaults(kwargs)
        n_chain = len(init_states)
        rngs = _get_per_chain_rngs(self.rng, n_chain)
        chain_iterators = _construct_chain_iterators(
            n_iter, kwargs.pop("progress_bar_class"), n_chain
        )
        if n_process == 1:
            # Using single process therefore run chains sequentially
            kwargs.pop("max_threads_per_process", None)
            final_states, traces, stats, adapter_states, _ = _sample_chains_sequential(
                init_states=init_states,
                rngs=rngs,
                chain_iterators=chain_iterators,
                transitions=self.transitions,
                trace_funcs=trace_funcs,
                **kwargs,
            )
        else:
            # Run chains in parallel using a multiprocess(ing).Pool
            final_states, traces, stats, adapter_states, _ = _sample_chains_parallel(
                init_states=init_states,
                rngs=rngs,
                chain_iterators=chain_iterators,
                transitions=self.transitions,
                trace_funcs=trace_funcs,
                n_process=n_process,
                **kwargs,
            )
        if len(adapter_states) > 0:
            _finalize_adapters(
                adapter_states, final_states, kwargs["adapters"], self.transitions, rngs
            )
        return final_states, traces, stats

    def sample_chains_with_adaptive_warm_up(
        self,
        n_warm_up_iter,
        n_main_iter,
        init_states,
        trace_funcs,
        adapters,
        stager=None,
        n_process=1,
        **kwargs,
    ):
        """Sample Markov chains from given initial states with adaptive warm up.

        One or more Markov chains are sampled, with each chain iteration
        consisting of one or more Markov transitions. The chains are split into
        multiple *stages* with one or more adaptive warm up stages followed by
        the main non-adaptive sampling stage. During the adaptive stage(s)
        parameters of the transition(s) are adaptively tuned based on the chain
        state and/or transition statistics.

        The chains (including both adaptive and non-adaptive stages) may be run
        in parallel across multiple independent processes or sequentially. In
        all cases all chains use independent random draws.

        Args:
            n_warm_up_iter (int): Number of adaptive warm up iterations per
                chain. Depending on the `mici.stagers.Stager` instance specified
                by the `stage arguments the warm up iterations may be split
                between one or more adaptive stages.
            n_main_iter (int): Number of iterations (samples to draw) per chain
                during main (non-adaptive) sampling stage.
            init_states (Iterable[ChainState] or Iterable[Dict[str, object]]):
                Initial chain states. Each entry can be either a `ChainState`
                object or a dictionary with entries specifying initial values
                for all state variables used by chain transition `sample`
                methods.
            trace_funcs (Iterable[Callable[[ChainState], Dict[str, array]]]):
                List of functions which compute the variables to be recorded at
                each chain iteration during the main non-adaptive sampling
                stage, with each trace function being passed the current state
                and returning a dictionary of scalar or array values
                corresponding to the variable(s) to be stored. The keys in the
                returned dictionaries are used to index the trace arrays in the
                returned traces dictionary. If a key appears in multiple
                dictionaries only the the value corresponding to the last trace
                function to return that key will be stored.
            adapters (Dict[str, Iterable[Adapter]): Dictionary of iterables
                of `mici.adapters.Adapter` instances keyed by strings
                corresponding to the key of the transition in the `transitions`
                dictionary to apply the adapters to, to use to adaptatively set
                parameters of the transitions during the adaptive stages of  the
                chains. Note that the adapter updates are applied in the order
                the adapters appear in the iterables and so if multiple adapters
                change the same parameter(s) the order will matter.

        Kwargs:
            stager (mici.stagers.Stager or None): Chain iteration stager object
                which controls the split of the chain iterations into the
                adaptive warm up and non-adaptive main stages. If set to `None`
                (the default) and all adapters specified by the `adapters`
                argument are of the fast type (i.e. their `is_fast` attribute is
                `True`) then a `mici.stagers.WarmUpStager` instance will be used
                corresponding to using a single adaptive warm up stage will all
                adapters active. If set to `None` and the adapters specified by
                the adapters argument are not all of the fast type, then a
                `mici.stagers.WindowedWarmUpStager` (with its default arguments)
                will be used, corresponding to using multiple adaptive warm up
                stages with only the fast-type adapters active in some - see
                docstring of `mici.stagers.WarmUpStager` for details.
            n_process (int or None): Number of parallel processes to run chains
                over. If `n_process=1` then chains will be run sequentially
                otherwise a `multiprocessing.Pool` object will be used to
                dynamically assign the chains across multiple processes. If set
                to `None` then the number of processes will be set to the
                output of `os.cpu_count()`. Default is `n_process=1`.
            max_threads_per_process (int or None): If `threadpoolctl` is
                available this argument may be used to limit the maximum number
                of threads that can be used in thread pools used in libraries
                supported by `threadpoolctl`, which include BLAS and OpenMP
                implementations. This argument will only have an effect if
                `n_process > 1` such that chains are being run on multiple
                processes and only if `threadpoolctl` is installed in the
                current Python environment. If set to `None` (the default) no
                limits are set.
            memmap_enabled (bool): Whether to memory-map arrays used to store
                chain data to files on disk to avoid excessive system memory
                usage for long chains and/or large chain states. The chain data
                is written to `.npy` files in the directory specified by
                `memmap_path` (or a temporary directory if not provided). These
                files persist after the termination of the function so should
                be manually deleted when no longer required. Default is to
                for memory mapping to be disabled.
            memmap_path (str): Path to directory to write memory-mapped chain
                data to. If not provided, a temporary directory will be created
                and the chain data written to files there.
            monitor_stats (Iterable[Tuple[str, str]]): List of tuples of string
                key pairs, with first entry the key of a Markov transition in
                the `transitions` dict passed to the the `__init__` method and
                the second entry the key of a chain statistic that will be
                returned in the `chain_stats` dictionary. The mean over samples
                computed so far of the chain statistics associated with any
                valid key-pairs will be monitored during sampling  by printing
                as postfix to progress bar.
            display_progress (bool): Whether to display a progress bar to
                track the completed chain sampling iterations. Default value
                is `True`, i.e. to display progress bar.
            progress_bar_class (mici.progressbars.BaseProgressBar): Class or
                factory function for progress bar to use to show chain
                progress if enabled (`display_progress=True`). Defaults to
                `mici.progressbars.ProgressBar`.

        Returns:
            final_states (List[ChainState]): States of chains after final
                iteration. May be used to resume sampling a chain by passing as
                the initial states to a new `sample_chains` call.
            traces (Dict[str, List[array]]): Dictionary of chain trace arrays.
                Values in dictionary are list of arrays of variables outputted
                by trace functions in `trace_funcs` with each array in the list
                corresponding to a single chain and the leading dimension of
                each array corresponding to the iteration (draw) index in the
                main non-adaptive sampling stage. The key for each value is the
                corresponding key in the dictionary returned by the trace
                function which computed the traced value.
            chain_stats (Dict[str, Dict[str, List[array]]]): Dictionary of
                chain transition statistic dictionaries. Values in outer
                dictionary are dictionaries of statistics for each chain
                transition, keyed by the string key for the transition. The
                values in each inner transition dictionary are lists of arrays
                of chain statistic values with each array in the list
                corresponding to a single chain and the leading dimension of
                each array corresponding to the iteration (draw) index in the
                main non-adaptive sampling stage. The key for each value is a
                string description of the corresponding integration transition
                statistic.
        """
        self.__set_sample_chain_kwargs_defaults(kwargs)
        n_chain = len(init_states)
        rngs = _get_per_chain_rngs(self.rng, n_chain)
        progress_bar_class = kwargs.pop("progress_bar_class")
        common_sample_chains_kwargs = {
            "rngs": rngs,
            "transitions": self.transitions,
            **kwargs,
        }
        if n_process > 1:
            sample_chains_func = _sample_chains_parallel
            common_sample_chains_kwargs["n_process"] = n_process
        else:
            sample_chains_func = _sample_chains_sequential
            common_sample_chains_kwargs.pop("max_threads_per_process", None)
        if stager is None:
            if all(a.is_fast for a_list in adapters.values() for a in a_list):
                stager = WarmUpStager()
            else:
                stager = WindowedWarmUpStager()
        sampling_stages = stager.stages(
            n_warm_up_iter, n_main_iter, adapters, trace_funcs
        )
        chain_states = init_states
        with LabelledSequenceProgressBar(
            sampling_stages, "Sampling stage", position=(0, n_chain + 1)
        ) as sampling_stages_pb:
            chain_iterators = _construct_chain_iterators(
                n_warm_up_iter, progress_bar_class, n_chain, 1
            )
            for stage, _ in sampling_stages_pb:
                for chain_it in chain_iterators:
                    chain_it.sequence = range(stage.n_iter)
                (
                    chain_states,
                    traces,
                    stats,
                    adapter_states,
                    exception,
                ) = sample_chains_func(
                    init_states=chain_states,
                    trace_funcs=stage.trace_funcs,
                    adapters=stage.adapters,
                    chain_iterators=chain_iterators,
                    **common_sample_chains_kwargs,
                )
                if len(adapter_states) > 0:
                    _finalize_adapters(
                        adapter_states,
                        chain_states,
                        stage.adapters,
                        self.transitions,
                        rngs,
                    )
                if isinstance(exception, KeyboardInterrupt):
                    return chain_states, traces, stats
        return chain_states, traces, stats


class HamiltonianMCMC(MarkovChainMonteCarloMethod):
    """Wrapper class for Hamiltonian Markov chain Monte Carlo (H-MCMC) methods.

    Here H-MCMC is defined as a MCMC method which augments the original target
    variable (henceforth position variable) with a momentum variable with a
    user specified conditional distribution given the position variable. In
    each chain iteration two Markov transitions leaving the resulting joint
    distribution on position and momentum variables invariant are applied -
    the momentum variables are updated in a transition which leaves their
    conditional distribution invariant (momentum transition) and then a
    trajectory in the joint space is generated by numerically integrating a
    Hamiltonian dynamic with an appropriate symplectic integrator which is
    exactly reversible, volume preserving and approximately conserves the joint
    probability density of the target-momentum state pair; one state from the
    resulting trajectory is then selected as the next joint chain state using
    an appropriate sampling scheme such that the joint distribution is left
    exactly invariant (integration transition).

    There are various options available for both the momentum transition and
    integration transition, with by default the momentum transition set to be
    independent resampling of the momentum variables from their conditional
    distribution.

    References:

      1. Duane, S., Kennedy, A.D., Pendleton, B.J. and Roweth, D., 1987.
         Hybrid Monte Carlo. Physics letters B, 195(2), pp.216-222.
      2. Neal, R.M., 2011. MCMC using Hamiltonian dynamics.
         Handbook of Markov Chain Monte Carlo, 2(11), p.2.
    """

    def __init__(self, system, rng, integration_transition, momentum_transition=None):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            rng (numpy.random.Generator): Numpy random number generator.
            integration_transition (mici.transitions.IntegrationTransition):
                Markov transition kernel which leaves canonical distribution
                invariant and jointly updates the position and momentum
                components of the chain state by integrating the Hamiltonian
                dynamics of the system to propose new values for the state.
            momentum_transition (None or mici.transitions.MomentumTransition):
                Markov transition kernel which leaves the conditional
                distribution on the momentum under the canonical distribution
                invariant, updating only the momentum component of the chain
                state. If set to `None` the momentum transition operator
                `mici.transitions.IndependentMomentumTransition` will be used,
                which independently samples the momentum from its conditional
                distribution.
        """
        self.system = system
        self.rng = rng
        if momentum_transition is None:
            momentum_transition = trans.IndependentMomentumTransition(system)
        super().__init__(
            rng,
            OrderedDict(
                momentum_transition=momentum_transition,
                integration_transition=integration_transition,
            ),
        )

    def _preprocess_init_state(self, init_state):
        """Make sure initial state is a ChainState and has momentum."""
        if isinstance(init_state, np.ndarray):
            # If array use to set position component of new ChainState
            init_state = ChainState(
                pos=init_state, 
                mom=None, 
                dir=1,
                constr_eval = [])
        elif not isinstance(init_state, ChainState) or "mom" not in init_state:
            raise TypeError(
                "init_state should be an array or `ChainState` with " "`mom` attribute."
            )
        if init_state.mom is None:
            init_state.mom = self.system.sample_momentum(init_state, self.rng)
        return init_state

    def _default_trace_func(self, state):
        """Default function of the chain state traced while sampling."""
        # This needs to be a method rather than for example a local nested
        # function in the __set_sample_chain_kwargs_defaults method to ensure
        # that it remains pickleable and so can be piped to a separate process
        # when running multiple chains using multiprocessing
        return {"pos": state.pos, "hamiltonian": self.system.h(state)}

    def __set_sample_chain_kwargs_defaults(self, kwargs):
        # default to tracing position component of state and Hamiltonian
        if "trace_funcs" not in kwargs:
            kwargs["trace_funcs"] = [self._default_trace_func]
        # if `monitor_stats` specified, expand all statistics keys to key pairs
        # with transition key set to `integration_transition`
        if "monitor_stats" in kwargs:
            kwargs["monitor_stats"] = [
                ("integration_transition", stats_key)
                for stats_key in kwargs["monitor_stats"]
            ]
        else:
            kwargs["monitor_stats"] = [("integration_transition", "accept_stat")]
        # if adapters kwarg specified, wrap adapter list in dictionary with
        # adapters applied to integration transition
        if "adapters" in kwargs and kwargs["adapters"] is not None:
            kwargs["adapters"] = {"integration_transition": kwargs["adapters"]}

    def sample_chain(self, n_iter, init_state, **kwargs):
        """Sample a Markov chain from a given initial state.

        Performs a specified number of chain iterations (each of which may be
        composed of multiple individual Markov transitions), recording the
        outputs of functions of the sampled chain state after each iteration.

        Args:
            n_iter (int): Number of iterations (samples to draw) per chain.
            init_state (mici.states.ChainState or array): Initial chain state.
                The state can be either an array specifying the state position
                component or a `mici.states.ChainState` instance. If an array
                is passed or the `mom` attribute of the state is not set, a
                momentum component will be independently sampled from its
                conditional distribution.

        Kwargs:
            trace_funcs (Iterable[Callable[[ChainState], Dict[str, array ]]]):
                List of functions which compute the variables to be recorded at
                each chain iteration, with each trace function being passed the
                current state and returning a dictionary of scalar or array
                values corresponding to the variable(s) to be stored. The keys
                in the returned dictionaries are used to index the trace arrays
                in the returned traces dictionary. If a key appears in multiple
                dictionaries only the the value corresponding to the last trace
                function to return that key will be stored. Default is to use a
                single function which recordes the position component of the
                state under the key `pos` and the Hamiltonian at the state
                under the key `hamiltonian`.
            memmap_enabled (bool): Whether to memory-map arrays used to store
                chain data to files on disk to avoid excessive system memory
                usage for long chains and/or large chain states. The chain data
                is written to `.npy` files in the directory specified by
                `memmap_path` (or a temporary directory if not provided). These
                files persist after the termination of the function so should
                be manually deleted when no longer required. Default is for
                memory mapping to be disabled.
            memmap_path (str): Path to directory to write memory-mapped chain
                data to. If not provided, a temporary directory will be created
                and the chain data written to files there.
            monitor_stats (Iterable[str]): List of string keys of chain
                statistics to monitor mean of over samples computed so far
                during sampling by printing as postfix to progress bar. Default
                is to print only the mean `accept_stat` statistic.
            display_progress (bool): Whether to display a progress bar to
                track the completed chain sampling iterations. Default value
                is `True`, i.e. to display progress bar.
            progress_bar_class (mici.progressbars.BaseProgressBar): Class or
                factory function for progress bar to use to show chain
                progress if enabled (`display_progress=True`). Defaults to
                `mici.progressbars.ProgressBar`.
            adapters (Iterable[Adapter]): Sequence of `mici.adapters.Adapter`
                instances to use to adaptatively set parameters of the
                integration transition such as the step size while sampling a
                chain. Note that the adapter updates are applied in the order
                the adapters appear in the iterable and so if multiple adapters
                change the same parameter(s) the order will matter. Adaptation
                based on the chain state history breaks the Markov property and
                so any chain samples while adaptation is active should not be
                used in estimates of expectations. Default is to use no
                adapters.

        Returns:
            final_state (mici.states.ChainState): State of chain after final
                iteration. May be used to resume sampling a chain by passing as
                the initial state to a new `sample_chain` call.
            traces (Dict[str, array]): Dictionary of chain trace arrays. Values
                in dictionary are arrays of variables outputted by trace
                functions in `trace_funcs` with leading dimension of array
                corresponding to the sampling (draw) index. The key for each
                value is the corresponding key in the dictionary returned by
                the trace function which computed the traced value.
            chain_stats (Dict[str, array]): Dictionary of chain integration
                transition statistics. Values in dictionary are arrays of chain
                statistic values with the leading dimension of each array
                corresponding to the sampling (draw) index. The key for each
                value is a string description of the corresponding integration
                transition statistic.
        """
        init_state = self._preprocess_init_state(init_state)
        self.__set_sample_chain_kwargs_defaults(kwargs)
        final_state, traces, chain_stats = super().sample_chain(
            n_iter, init_state, **kwargs
        )
        chain_stats = chain_stats.get("integration_transition", {})
        return final_state, traces, chain_stats

    def sample_chains(self, n_iter, init_states, **kwargs):
        """Sample one or more Markov chains from given initial states.

        Performs a specified number of chain iterations, each of consisting of a
        momentum transition followed by an integration transition, recording the
        outputs of functions of the sampled chain state after each iteration.
        The chains may be run in parallel across multiple independent processes
        or sequentially. In all cases all chains use independent random draws.

        Args:
            n_iter (int): Number of iterations (samples to draw) per chain.
            init_states (Iterable[ChainState] or Iterable[array]): Initial
                chain states. Each state can be either an array specifying the
                state position component or a `mici.states.ChainState`
                instance. If an array is passed or the `mom` attribute of the
                state is not set, a momentum component will be independently
                sampled from its conditional distribution. One chain will be
                run for each state in the iterable sequence.

        Kwargs:
            n_process (int or None): Number of parallel processes to run chains
                over. If `n_process=1` then chains will be run sequentially
                otherwise a `multiprocessing.Pool` object will be used to
                dynamically assign the chains across multiple processes. If set
                to `None` then the number of processes will be set to the
                output of `os.cpu_count()`. Default is `n_process=1`.
            max_threads_per_process (int or None): If `threadpoolctl` is
                available this argument may be used to limit the maximum number
                of threads that can be used in thread pools used in libraries
                supported by `threadpoolctl`, which include BLAS and OpenMP
                implementations. This argument will only have an effect if
                `n_process > 1` such that chains are being run on multiple
                processes and only if `threadpoolctl` is installed in the
                current Python environment. If set to `None` (the default) no
                limits are set.
            trace_funcs (Iterable[Callable[[ChainState], Dict[str, array]]]):
                List of functions which compute the variables to be recorded at
                each chain iteration, with each trace function being passed the
                current state and returning a dictionary of scalar or array
                values corresponding to the variable(s) to be stored. The keys
                in the returned dictionaries are used to index the trace arrays
                in the returned traces dictionary. If a key appears in multiple
                dictionaries only the the value corresponding to the last trace
                function to return that key will be stored.  Default is to use a
                single function which recordes the position component of the
                state under the key `pos` and the Hamiltonian at the state
                under the key `hamiltonian`.
            memmap_enabled (bool): Whether to memory-map arrays used to store
                chain data to files on disk to avoid excessive system memory
                usage for long chains and/or large chain states. The chain data
                is written to `.npy` files in the directory specified by
                `memmap_path` (or a temporary directory if not provided). These
                files persist after the termination of the function so should
                be manually deleted when no longer required. Default is for
                memory mapping to be disabled.
            memmap_path (str): Path to directory to write memory-mapped chain
                data to. If not provided (the default), a temporary directory
                will be created and the chain data written to files there.
            monitor_stats (Iterable[str]): List of string keys of chain
                statistics to monitor mean of over samples computed so far
                during sampling by printing as postfix to progress bar. Default
                is to print only the mean `accept_stat` statistic.
            display_progress (bool): Whether to display a progress bar to
                track the completed chain sampling iterations. Default value
                is `True`, i.e. to display progress bar.
            progress_bar_class (mici.progressbars.BaseProgressBar): Class or
                factory function for progress bar to use to show chain
                progress if enabled (`display_progress=True`). Defaults to
                `mici.progressbars.ProgressBar`.
            adapters (Iterable[Adapter]): Sequence of `mici.adapters.Adapter`
                instances to use to adaptatively set parameters of the
                integration transition such as the step size while sampling a
                chain. Note that the adapter updates are applied in the order
                the adapters appear in the iterable and so if multiple adapters
                change the same parameter(s) the order will matter. Adaptation
                based on the chain state history breaks the Markov property and
                so any chain samples while adaptation is active should not be
                used in estimates of expectations. Default is to use no
                adapters.

        Returns:
            final_states (List[ChainState]): States of chains after final
                iteration. May be used to resume sampling a chain by passing as
                the initial states to a new `sample_chains` call.
            traces (Dict[str, List[array]]): Dictionary of chain trace arrays.
                Values in dictionary are list of arrays of variables outputted
                by trace functions in `trace_funcs` with each array in the list
                corresponding to a single chain and the leading dimension of
                each array corresponding to the iteration (draw) index. The key
                for each value is the corresponding key in the dictionary
                returned by the trace function which computed the traced value.
            chain_stats (Dict[str, List[array]]): Dictionary of chain
                integration transition statistics. Values in dictionary are
                lists of arrays of chain statistic values with each array in
                the list corresponding to a single chain and the leading
                dimension of each array corresponding to the iteration (draw)
                index. The key for each value is a string description of the
                corresponding integration transition statistic.
        """
        init_states = [self._preprocess_init_state(i) for i in init_states]
        self.__set_sample_chain_kwargs_defaults(kwargs)
        final_states, traces, chain_stats = super().sample_chains(
            n_iter, init_states, **kwargs
        )
        chain_stats = chain_stats.get("integration_transition", {})
        return final_states, traces, chain_stats

    def sample_chains_with_adaptive_warm_up(
        self, n_warm_up_iter, n_main_iter, init_states, **kwargs
    ):
        """Sample Markov chains from given initial states with adaptive warm up.

        One or more Markov chains are sampled, with each chain iteration
        consisting of a momentum transition followed by an integration
        transition. The chains are split into multiple *stages* with one or more
        adaptive warm up stages followed by the main non-adaptive sampling
        stage. During the adaptive stage(s) parameters of the integration
        transition such as the integrator step size are adaptively tuned based
        on the chain state and/or transition statistics.

        The default settings use a single (fast) `DualAveragingStepSizeAdapter`
        adapter instance which adapts the integrator step-size using a
        dual-averaging algorithm in a single adaptive stage.

        The chains (including both adaptive and non-adaptive stages) may be run
        in parallel across multiple independent processes or sequentially. In
        all cases all chains use independent random draws.

        Args:
            n_warm_up_iter (int): Number of adaptive warm up iterations per
                chain. Depending on the `mici.stagers.Stager` instance specified
                by the `stage arguments the warm up iterations may be split
                between one or more adaptive stages.
            n_main_iter (int): Number of iterations (samples to draw) per chain
                during main (non-adaptive) sampling stage.
            init_states (Iterable[ChainState] or Iterable[array]): Initial
                chain states. Each state can be either an array specifying the
                state position component or a `mici.states.ChainState`
                instance. If an array is passed or the `mom` attribute of the
                state is not set, a momentum component will be independently
                sampled from its conditional distribution. One chain will be
                run for each state in the iterable sequence.

        Kwargs:
            n_process (int or None): Number of parallel processes to run chains
                over. If `n_process=1` then chains will be run sequentially
                otherwise a `multiprocessing.Pool` object will be used to
                dynamically assign the chains across multiple processes. If set
                to `None` then the number of processes will be set to the
                output of `os.cpu_count()`. Default is `n_process=1`.
            max_threads_per_process (int or None): If `threadpoolctl` is
                available this argument may be used to limit the maximum number
                of threads that can be used in thread pools used in libraries
                supported by `threadpoolctl`, which include BLAS and OpenMP
                implementations. This argument will only have an effect if
                `n_process > 1` such that chains are being run on multiple
                processes and only if `threadpoolctl` is installed in the
                current Python environment. If set to `None` (the default) no
                limits are set.
            trace_funcs (Iterable[Callable[[ChainState], Dict[str, array]]]):
                List of functions which compute the variables to be recorded at
                each chain iteration during the main non-adaptive sampling
                stage, with each trace function being passed the current state
                and returning a dictionary of scalar or array values
                corresponding to the variable(s) to be stored. The keys in the
                returned dictionaries are used to index the trace arrays in the
                returned traces dictionary. If a key appears in multiple
                dictionaries only the the value corresponding to the last trace
                function to return that key will be stored.  Default is to use a
                single function which recordes the position component of the
                state under the key `pos` and the Hamiltonian at the state under
                the key `hamiltonian`.
            adapters (Iterable[Adapter]): List of `mici.adapters.Adapter`
                instances to use to adaptatively set parameters of the
                integration transition such as the step size during the adaptive
                stages of the chains. Note that the adapter updates are applied
                in the order the adapters appear in the iterable and so if
                multiple adapters change the same parameter(s) the order will
                matter. Default is to use a single instance of
                `mici.adapters.DualAveragingStepSizeAdapter` with its default
                parameters.
            stager (mici.stagers.Stager or None): Chain iteration stager object
                which controls the split of the chain iterations into the
                adaptive warm up and non-adaptive main stages. If set to `None`
                (the default) and all adapters specified by the `adapters`
                argument are of the fast type (i.e. their `is_fast` attribute is
                `True`) then a `mici.stagers.WarmUpStager` instance will be used
                corresponding to using a single adaptive warm up stage will all
                adapters active. If set to `None` and the adapters specified by
                the adapters argument are not all of the fast type, then a
                `mici.stagers.WindowedWarmUpStager` (with its default arguments)
                will be used, corresponding to using multiple adaptive warm up
                stages with only the fast-type adapters active in some - see
                docstring of `mici.stagers.WarmUpStager` for details.
            memmap_enabled (bool): Whether to memory-map arrays used to store
                chain data to files on disk to avoid excessive system memory
                usage for long chains and/or large chain states. The chain data
                is written to `.npy` files in the directory specified by
                `memmap_path` (or a temporary directory if not provided). These
                files persist after the termination of the function so should
                be manually deleted when no longer required. Default is for
                memory mapping to be disabled.
            memmap_path (str): Path to directory to write memory-mapped chain
                data to. If not provided (the default), a temporary directory
                will be created and the chain data written to files there.
            monitor_stats (Iterable[str]): List of string keys of chain
                statistics to monitor mean of over samples computed so far
                during sampling by printing as postfix to progress bar. Default
                is to print only the mean `accept_stat` statistic.
            display_progress (bool): Whether to display a progress bar to
                track the completed chain sampling iterations. Default value
                is `True`, i.e. to display progress bar.
            progress_bar_class (mici.progressbars.BaseProgressBar): Class or
                factory function for progress bar to use to show chain
                progress if enabled (`display_progress=True`). Defaults to
                `mici.progressbars.ProgressBar`.

        Returns:
            final_states (List[ChainState]): States of chains after final
                iteration. May be used to resume sampling a chain by passing as
                the initial states to a new `sample_chains` call.
            traces (Dict[str, List[array]]): Dictionary of chain trace arrays.
                Values in dictionary are list of arrays of variables outputted
                by trace functions in `trace_funcs` with each array in the list
                corresponding to a single chain and the leading dimension of
                each array corresponding to the iteration (draw) index within
                the main non-adaptive sampling stage. The key for each value is
                the corresponding key in the dictionary returned by the trace
                function which computed the traced value.
            chain_stats (Dict[str, List[array]]): Dictionary of chain
                integration transition statistics. Values in dictionary are
                lists of arrays of chain statistic values with each array in the
                list corresponding to a single chain and the leading dimension
                of each array corresponding to the iteration (draw) index within
                the main non-adaptive sampling stage. The key for each value is
                a string description of the corresponding integration transition
                statistic.
        """
        init_states = [self._preprocess_init_state(i) for i in init_states]
        if "adapters" not in kwargs:
            kwargs["adapters"] = [DualAveragingStepSizeAdapter()]
        self.__set_sample_chain_kwargs_defaults(kwargs)
        final_states, traces, chain_stats = super().sample_chains_with_adaptive_warm_up(
            n_warm_up_iter, n_main_iter, init_states, **kwargs
        )
        chain_stats = chain_stats.get("integration_transition", {})
        return final_states, traces, chain_stats


class StaticMetropolisHMC(HamiltonianMCMC):
    """Static integration time H-MCMC implementation with Metropolis sampling.

    In each transition a trajectory is generated by integrating the Hamiltonian
    dynamics from the current state in the current integration time direction
    for a fixed integer number of integrator steps.

    The state at the end of the trajectory with the integration direction
    negated (this ensuring the proposed move is an involution) is used as the
    proposal in a Metropolis acceptance step. The integration direction is then
    deterministically negated again irrespective of the accept decision, with
    the effect being that on acceptance the integration direction will be equal
    to its initial value and on rejection the integration direction will be
    the negation of its initial value.

    This is original proposed Hybrid Monte Carlo (often now instead termed
    Hamiltonian Monte Carlo) algorithm [1, 2].

    References:

      1. Duane, S., Kennedy, A.D., Pendleton, B.J. and Roweth, D., 1987.
         Hybrid Monte Carlo. Physics letters B, 195(2), pp.216-222.
      2. Neal, R.M., 2011. MCMC using Hamiltonian dynamics.
         Handbook of Markov Chain Monte Carlo, 2(11), p.2.
    """

    def __init__(self, system, integrator, rng, n_step, momentum_transition=None):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            rng (numpy.random.Generator): Numpy random number generator.
            integrator (mici.integrators.Integrator): Symplectic integrator to
                use to simulate dynamics in integration transition.
            n_step (int): Number of integrator steps to simulate in each
                integration transition.
            momentum_transition (None or mici.transitions.MomentumTransition):
                Markov transition kernel which leaves the conditional
                distribution on the momentum under the canonical distribution
                invariant, updating only the momentum component of the chain
                state. If set to `None` the momentum transition operator
                `mici.transitions.IndependentMomentumTransition` will be used,
                which independently samples the momentum from its conditional
                distribution.
        """
        integration_transition = trans.MetropolisStaticIntegrationTransition(
            system, integrator, n_step
        )
        super().__init__(system, rng, integration_transition, momentum_transition)

    @property
    def n_step(self):
        """Number of integrator steps per integrator transition."""
        return self.transitions["integration_transition"].n_step

    @n_step.setter
    def n_step(self, value):
        assert value > 0, "n_step must be non-negative"
        self.transitions["integration_transition"].n_step = value


class DynamicMultinomialHMC(HamiltonianMCMC):
    """Dynamic integration time H-MCMC with multinomial sampling of new state.

    In each transition a binary tree of states is recursively computed by
    integrating randomly forward and backward in time by a number of steps
    equal to the previous tree size [1, 2] until a termination criteria on the
    tree leaves is met. The next chain state is chosen from the candidate
    states using a progressive multinomial sampling scheme [2] based on the
    relative probability densities of the different candidate states, with the
    resampling biased towards states further from the current state.

    When used with the default settings of `riemannian_no_u_turn_criterion`
    termination criterion and extra subtree checks enabled, this sampler is
    equivalent to the default 'NUTS' MCMC algorithm (minus adaptation) used in
    [Stan](https://mc-stan.org/) as of version v2.23.

    References:

      1. Hoffman, M.D. and Gelman, A., 2014. The No-U-turn sampler:
         adaptively setting path lengths in Hamiltonian Monte Carlo.
         Journal of Machine Learning Research, 15(1), pp.1593-1623.
      2. Betancourt, M., 2017. A conceptual introduction to Hamiltonian Monte
         Carlo. arXiv preprint arXiv:1701.02434.
    """

    def __init__(
        self,
        system,
        integrator,
        rng,
        max_tree_depth=10,
        max_delta_h=1000,
        termination_criterion=trans.riemannian_no_u_turn_criterion,
        do_extra_subtree_checks=True,
        momentum_transition=None,
    ):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            rng (numpy.random.Generator): Numpy random number generator.
            integrator (mici.integrators.Integrator): Symplectic integrator to
                use to simulate dynamics in integration transition.
            max_tree_depth (int): Maximum depth to expand trajectory binary
                tree to in integrator transition. The maximum number of
                integrator steps corresponds to `2**max_tree_depth`.
            max_delta_h (float): Maximum change to tolerate in the Hamiltonian
                function over a trajectory in integrator transition before
                signalling a divergence.
            termination_criterion (
                    Callable[[System, ChainState, ChainState, array], bool]):
                Function computing criterion to use to determine when to
                terminate trajectory tree expansion. The function should take a
                Hamiltonian system as its first argument, a pair of states
                corresponding to the two edge nodes in the trajectory
                (sub-)tree being checked and an array containing the sum of the
                momentums over the trajectory (sub)-tree. Defaults to
                `mici.transitions.riemannian_no_u_turn_criterion`.
            do_extra_subtree_checks (bool): Whether to perform additional
                termination criterion checks on overlapping subtrees of the
                current tree to improve robustness in systems with dynamics
                which are well approximated by independent system of simple
                harmonic oscillators. In such systems (corresponding to e.g.
                a standard normal target distribution and identity metric
                matrix representation) at certain step sizes a 'resonant'
                behaviour is seen by which the termination criterion fails to
                detect that the trajectory has expanded past a half-period i.e.
                has 'U-turned' resulting in trajectories continuing to expand,
                potentially up until the `max_tree_depth` limit is hit. For more
                details see [this Stan Discourse discussion](kutt.it/yAkIES).
                If `do_extra_subtree_checks` is set to `True` additional
                termination criterion checks are performed on overlapping
                subtrees which help to reduce this resonant behaviour at the
                cost of more conservative trajectory termination in some
                correlated models and some overhead from additional checks.
            momentum_transition (None or mici.transitions.MomentumTransition):
                Markov transition kernel which leaves the conditional
                distribution on the momentum under the canonical distribution
                invariant, updating only the momentum component of the chain
                state. If set to `None` the momentum transition operator
                `mici.transitions.IndependentMomentumTransition` will be used,
                which independently samples the momentum from its conditional
                distribution.
        """
        integration_transition = trans.MultinomialDynamicIntegrationTransition(
            system,
            integrator,
            max_tree_depth,
            max_delta_h,
            termination_criterion,
            do_extra_subtree_checks,
        )
        super().__init__(system, rng, integration_transition, momentum_transition)

    @property
    def max_tree_depth(self):
        """Maximum depth to expand trajectory binary tree to."""
        return self.transitions["integration_transition"].max_tree_depth

    @max_tree_depth.setter
    def max_tree_depth(self, value):
        assert value > 0, "max_tree_depth must be non-negative"
        self.transitions["integration_transition"].max_tree_depth = value

    @property
    def max_delta_h(self):
        """Change in Hamiltonian over trajectory to trigger divergence."""
        return self.transitions["integration_transition"].max_delta_h

    @max_delta_h.setter
    def max_delta_h(self, value):
        self.transitions["integration_transition"].max_delta_h = value
