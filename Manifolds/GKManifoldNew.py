"""
Newer version of Manifold for G-and-K problem.
"""
import numpy as np
from numpy import r_, exp, log, vstack, eye, prod, zeros, isfinite, ones, diag, pi
from numpy.linalg import norm
from numpy.random import default_rng, randn
from scipy.optimize import fsolve
from scipy.special import ndtri, ndtr
from scipy.linalg import block_diag
from scipy.stats import uniform as udist
from scipy.stats import norm as ndist
from scipy.stats import multivariate_normal as MVN
from warnings import catch_warnings, filterwarnings, resetwarnings

from autograd.numpy import concatenate as aconcatenate
from autograd.scipy.special import erf as aerf
from autograd.numpy import sqrt as asqrt
from autograd.numpy import exp as aexp
from autograd import jacobian

from Manifolds.Manifold import Manifold


class GKManifold(Manifold):
    def __init__(self, ystar, kernel_type='normal', use_autograd=False):
        assert kernel_type in ['normal', 'uniform']
        self.m = len(ystar)            # Number constraints = dimensionality of the data
        self.d = 4                     # Manifold has dimension 4 (like the parameter θ)
        self.n = self.d + self.m       # Dimension of ambient space is m + 4
        self.ystar = ystar
        self.kernel_type = kernel_type
        # N(0, 1) ---> U(0, 10).
        self.G    = lambda θ: 10*ndtr(θ)
        # U(0, 10) ---> N(0, 1)
        self.Ginv = lambda θ: ndtri(θ/10)
        # autograd jacobian
        self.fullJacobianAutograd = jacobian(self.q_autograd_raw)
        if use_autograd:
            self.fullJacobian = self.fullJacobianAutograd

    def q(self, ξ):
        """Constraint for G and K."""
        ξ = r_[self.G(ξ[:4]), ξ[4:]]   # expecting theta part to be N(0, 1)
        with catch_warnings():
            filterwarnings('error')
            try:
                return (ξ[0] + ξ[1]*(1 + 0.8*(1 - exp(-ξ[2]*ξ[4:]))/(1 + exp(-ξ[2]*ξ[4:]))) * ((1 + ξ[4:]**2)**ξ[3])*ξ[4:]) - self.ystar
            except RuntimeWarning:
                raise ValueError("Constraint found Overflow warning.")

    def _q_raw_uniform(self, ξ):
        """Constraint function expecting ξ[:4] ~ U(0, 10). It doesn't do any warning check."""
        return (ξ[0] + ξ[1]*(1 + 0.8*(1 - exp(-ξ[2]*ξ[4:]))/(1 + exp(-ξ[2]*ξ[4:]))) * ((1 + ξ[4:]**2)**ξ[3])*ξ[4:]) - self.ystar
    def _q_raw_normal(self, ξ):
        """Same as `_q_raw_uniform` except expects ξ[:4]~N(0,1)."""
        ξ = r_[self.G(ξ[:4]), ξ[4:]]
        return self._q_raw_uniform(ξ)

    def q_autograd_raw(self, ξ):
        """Raw version of the constraint function using autograd."""
        # transform using G
        ndtr_autograd = lambda θ: (aerf(θ/asqrt(2))+1)/2
        G_autograd = lambda θ: 10*ndtr_autograd(θ)
        ξ = aconcatenate((G_autograd(ξ[:4]), ξ[4:]))
        return (ξ[0] + ξ[1]*(1 + 0.8*(1 - aexp(-ξ[2]*ξ[4:]))/(1 + aexp(-ξ[2]*ξ[4:]))) * ((1 + ξ[4:]**2)**ξ[3])*ξ[4:]) - self.ystar

    def q_autograd(self, ξ):
        """Add the catch warnings thing."""
        with catch_warnings():
            filterwarnings('error')
            try:
                return self.q_autograd_raw(ξ)
            except RuntimeWarning:
                raise ValueError("Constraint found Overflow warning.")

    def Q(self, ξ):
        """Transpose of Jacobian for G and K. """
        ξ = r_[self.G(ξ[:4]), ξ[4:]]
        return vstack((
        ones(len(ξ[4:])),
        (1 + 0.8 * (1 - exp(-ξ[2] * ξ[4:])) / (1 + exp(-ξ[2] * ξ[4:]))) * ((1 + ξ[4:]**2)**ξ[3]) * ξ[4:],
        8 * ξ[1] * (ξ[4:]**2) * ((1 + ξ[4:]**2)**ξ[3]) * exp(ξ[2]*ξ[4:]) / (5 * (1 + exp(ξ[2]*ξ[4:]))**2),
        ξ[1]*ξ[4:]*((1+ξ[4:]**2)**ξ[3])*(1 + 9*exp(ξ[2]*ξ[4:]))*log(1 + ξ[4:]**2) / (5*(1 + exp(ξ[2]*ξ[4:]))),
        diag(ξ[1]*((1+ξ[4:]**2)**(ξ[3]-1))*(((18*ξ[3] + 9)*(ξ[4:]**2) + 9)*exp(2*ξ[2]*ξ[4:]) + (8*ξ[2]*ξ[4:]**3 + (20*ξ[3] + 10)*ξ[4:]**2 + 8*ξ[2]*ξ[4:] + 10)*exp(ξ[2]*ξ[4:]) + (2*ξ[3] + 1)*ξ[4:]**2 + 1) / (5*(1 + exp(ξ[2]*ξ[4:]))**2))
    ))

    def J(self, ξ):
        """Safely computes Jacobian."""
        with catch_warnings():
            filterwarnings('error')
            try:
                return self.Q(ξ).T
            except RuntimeWarning:
                raise ValueError("J computation found Runtime warning.")

    def fullJacobian(self, ξ):
        """J_f(G(ξ)) * J_G(ξ)."""
        JGbar = block_diag(10*np.diag(ndist.pdf(ξ[:4])), eye(len(ξ[4:])))
        return self.J(ξ) @ JGbar

    def log_parameter_prior(self, θ):
        """IMPORTANT: Typically the prior distribution is a U(0, 10) for all four parameters.
        We keep the same prior but since we don't want to work on a constrained space, we
        reparametrize the problem to an unconstrained space N(0, 1)."""
        with catch_warnings():
            filterwarnings('error')
            try:
                return udist.logpdf(self.G(θ), loc=0.0, scale=10.0).sum() + ndist.logpdf(θ).sum()
            except RuntimeWarning:
                return -np.inf

    def logprior(self, ξ):
        """Computes the prior distribution for G and K problem. Notice this is already reparametrized."""
        #return self.log_parameter_prior(ξ[:4]) - ξ[4:]@ξ[4:]/2
        return ndist.logpdf(ξ).sum()  # Should be the same as the commented code above

    def logη(self, ξ):
        """log posterior for c-rwm. This is on the manifold."""
        try:
            J = self.J(ξ)
            logprior = self.logprior(ξ)
            correction_term  = - prod(np.linalg.slogdet(J@J.T))/2
            return  logprior + correction_term
        except ValueError as e:
            return -np.inf

    def generate_logηϵ(self, ϵ):
        """Returns the log abc posterior for THUG."""
        if self.kernel_type not in ['normal', 'uniform']:
            raise NotImplementedError
        else:
            if self.kernel_type == 'normal':
                def log_abc_posterior(ξ):
                    """Log-ABC-posterior."""
                    u = self.q(ξ)
                    m = len(u)
                    return self.logprior(ξ) - u@u/(2*ϵ**2) - m*log(ϵ) - m*log(2*pi)/2
                return log_abc_posterior
            else:
                # uniform kernel
                def log_abc_posterior(ξ):
                    with np.errstate(divide='ignore'):
                        try:
                            return log(float(norm(self.q(ξ)) <= ϵ)) - self.m*log(ϵ)
                        except ValueError:
                            return -np.inf
                return log_abc_posterior

    def generate_logprior(self, ϵ):
        """Just used by markov_snippets.py"""
        assert ϵ == -np.inf, "ϵ must be -np.inf."
        return self.logprior


    def logp(self, v):
        """Log density for normal on the tangent space."""
        return MVN(mean=zeros(self.d), cov=eye(self.d)).logpdf(v)

    def is_on_manifold(self, ξ, tol=1e-8):
        """Checks if ξ is on the ystar manifold."""
        return np.max(abs(self.q(ξ))) < tol


"""
OTHER FUNCTIONS
"""

def data_generator(θ0, m, seed):
    """Stochastic Simulator. Generates y given θ."""
    rng = default_rng(seed)
    z = rng.normal(size=m)
    ξ = r_[θ0, z]
    return ξ[0] + ξ[1]*(1 + 0.8*(1 - exp(-ξ[2]*ξ[4:]))/(1 + exp(-ξ[2]*ξ[4:]))) * ((1 + ξ[4:]**2)**ξ[3])*ξ[4:]

def find_point_on_manifold(ystar, ϵ, max_iter=1000, tol=1.49012e-08, kernel_type='normal'):
    """Find a point on the data manifold."""
    i = 0
    manifold = GKManifold(ystar=ystar, kernel_type=kernel_type)
    log_abc_posterior = manifold.generate_logηϵ(ϵ)
    with catch_warnings():
        filterwarnings('error')
        while i <= max_iter:
            i += 1
            try:
                # Sample θ from U(0, 10)
                θfixed = randn(4)
                function = lambda z: manifold._q_raw_normal(r_[θfixed, z])
                z_guess  = randn(manifold.m)
                z_found  = fsolve(function, z_guess, xtol=tol)
                ξ_found  = r_[θfixed, z_found]
                if not isfinite([log_abc_posterior(ξ_found)]):
                    pass
                else:
                    resetwarnings()
                    return ξ_found

            except RuntimeWarning:
                continue
        resetwarnings()
        raise ValueError("Couldn't find a point, try again.")


def find_point_on_manifold_from_θ(ystar, θfixed, ϵ, maxiter=2000, tol=1.49012e-08, kernel_type='normal'):
    """Same as the above but we provide the θfixed. Can be used to find a point where
    the theta is already θ0."""
    i = 0
    manifold = GKManifold(ystar=ystar, kernel_type=kernel_type)
    log_abc_posterior = manifold.generate_logηϵ(ϵ)
    function = lambda z: manifold._q_raw_normal(r_[θfixed, z])
    with catch_warnings():
        filterwarnings('error')
        while i <= maxiter:
            i += 1
            try:
                z_guess  = randn(manifold.m)
                z_found  = fsolve(function, z_guess, xtol=tol)
                ξ_found  = r_[θfixed, z_found]
                if not isfinite([log_abc_posterior(ξ_found)]):
                    resetwarnings()
                    raise ValueError("Couldn't find a point.")
                else:
                    resetwarnings()
                    return ξ_found
            except RuntimeWarning:
                continue
        resetwarnings()
        raise ValueError("Couldn't find a point, try again.")
