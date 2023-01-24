# Experiment 39b: Same as experiment 39 but here we want to use Zappa (C-RWM) rather than C-HMC from mici.
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

import numpy as np
from numpy.linalg import norm, solve, det
from numpy.random import default_rng, rand, randn
from numpy import log, exp, zeros, eye, pi, array, diag, sqrt, cumsum, where, vstack, array
from numpy import apply_along_axis, outer, errstate, save, mean, sqrt, logspace, concatenate
from numpy import stack, cos, linspace, float64, finfo, full, nan
from scipy.stats import multivariate_normal as MVN
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time
from warnings import catch_warnings, filterwarnings
from itertools import product

from Manifolds.Manifold import Manifold
from utils import ESS_univariate
from Zappa.zappa import zappa_sampling_storecomps

import autograd.numpy as anp
import autograd.scipy as asp
import symnum.numpy as snp
import sympy
from symnum import (
    numpify, named_array, jacobian, grad, 
    vector_jacobian_product, matrix_hessian_product)


def split_into_integer_parts(n, m):
    return [round(n / m)] * (m - 1) + [n - round(n / m) * (m - 1)]

def grid_on_interval(interval, n_points, cosine_spacing=False):
    if cosine_spacing:
        # Use non-linear spacing with higher density near endpoints
        ts =  ((1 + cos(linspace(0, 1, n_points) * pi)) / 2)
    else:
        ts = linspace(0, 1, n_points)
    # If open interval space over range [left + eps, right - eps]
    eps = 10 * finfo(float64).eps
    left = (float(interval.left) + eps if interval.left_open 
            else float(interval.left))
    right = (float(interval.right) - eps if interval.right_open 
             else float(interval.right))
    return left + ts * (right - left)


def solve_for_limiting_manifold(y, n_points=200, cosine_spacing=False):
    assert n_points % 2 == 0, 'n_points must be even'
    θ = named_array('θ', 2)
    # solve F(θ) = y for θ[1] in terms of θ[0]
    θ_1_gvn_θ_0 = sympy.solve(forward_func(θ)[0] - y, θ[1])
    # find interval(s) over which θ[0] gives real θ[1] solutions
    θ_0_range = sympy.solveset(
        θ_1_gvn_θ_0[0]**2 > 0, θ[0], domain=sympy.Reals)
    θ_0_intervals = (
        θ_0_range.args if isinstance(θ_0_range, sympy.Union) 
        else [θ_0_range])
    # create  grid of values over valid θ[0] interval(s)
    n_intervals = len(θ_0_intervals)
    θ_0_grids = [
        grid_on_interval(intvl, n_pt + 1, cosine_spacing)
        for intvl, n_pt in zip(
            θ_0_intervals, 
            split_into_integer_parts(n_points // 2, n_intervals))]
    # generate NumPy function to calculate θ[1] in terms of θ[0]
    solve_func = sympy.lambdify(θ[0], θ_1_gvn_θ_0)
    manifold_points = []
    for θ_0_grid in θ_0_grids:
        # numerically calculate +/- θ[1] solutions over θ[0] grid
        θ_1_grid_neg, θ_1_grid_pos = solve_func(θ_0_grid)
        # stack θ[0] and θ[1] values in to 2D array in anticlockwise order
        manifold_points.append(stack([
            concatenate([θ_0_grid, θ_0_grid[-2:0:-1]]),
            concatenate([θ_1_grid_neg, θ_1_grid_pos[-2:0:-1]])
        ], -1))
    return manifold_points




if __name__ == '__main__':
    # Posterior Parameters
    σ = 0.1
    y = 1
    dim_θ = 2
    dim_y = 1

    # Functions
    @numpify(dim_θ)
    def forward_func(θ):
        return snp.array([θ[1]**2 + 3 * θ[0]**2 * (θ[0]**2 - 1)])

    @numpify(dim_θ + dim_y)
    def neg_log_prior_dens(q):
        return snp.sum(q**2) / 2

    @numpify(dim_θ, None, None)
    def neg_log_posterior_dens(θ, σ, y):
        return (snp.sum(θ**2, 0) + snp.sum((y - forward_func(θ))**2, 0) / σ**2) / 2

    @numpify(dim_θ, None)
    def metric(θ, σ):
        jac = jacobian(forward_func)(θ)
        return jac.T @ jac / σ**2 + snp.identity(dim_θ)

    @numpify(dim_θ, None, None, None)
    def neg_log_lifted_posterior_dens(θ, η, σ, y):
        jac = jacobian(forward_func)(θ)
        return snp.sum(θ**2, 0) / 2 + η**2 / 2 + snp.log(jac @ jac.T + σ**2)[0, 0] / 2

    def log_lifted_posterior_dens(xi, sigma):
        """for c-rwm"""
        jac = jacobian(forward_func)(xi[:2])
        return - snp.sum(xi[:2]**2, 0) / 2 - xi[-1]**2 / 2 - snp.log(jac @ jac.T + sigma**2)[0, 0] / 2

    @numpify(dim_θ + dim_y, None, None)
    def constr(q, σ, y):
        θ, η = q[:dim_θ], q[dim_θ:]
        return forward_func(θ) + σ * η - y

    # Settings
    n_sample = 50
    n_chain = 10
    n_step = 20

    # Auxiliary
    σ_grid = logspace(-5, 0, 11)
    ϵ_grid = logspace(-5, 0, 11)
    θ_inits = concatenate(solve_for_limiting_manifold(y, n_chain))

    # Storage
    crwm_av_accept_probs  = full((σ_grid.shape[0], ϵ_grid.shape[0]), 0.0)
    crwm_av_njacob_evals  = full((σ_grid.shape[0], ϵ_grid.shape[0]), 0.0)
    crwm_av_miness_runtime = full((σ_grid.shape[0], ϵ_grid.shape[0]), 0.0)


    # More functions
    seed = 20200310
    rng = default_rng(seed)
    grad_neg_log_posterior_dens = grad(neg_log_posterior_dens)
    grad_and_val_neg_log_posterior_dens = grad(neg_log_posterior_dens, return_aux=True)
    vjp_metric = vector_jacobian_product(metric, return_aux=True)
    grad_neg_log_prior_dens = grad(neg_log_prior_dens)
    jacob_constr = jacobian(constr, return_aux=True)
    mhp_constr = matrix_hessian_product(constr, return_aux=True)
    _ = np.seterr(invalid='ignore', over='ignore')


    # Run Hug and Thug on the posterior distribution
    q = MVN(zeros(2), eye(2))
    prior_log_dens = lambda x: MVN(zeros(2), eye(2)).logpdf(x)
    grad_log_prior = lambda x: -x
    F = lambda θ: array([θ[1]**2 + 3 * θ[0]**2 * (θ[0]**2 - 1)])
    grad_F = lambda θ: array([12*θ[0]**3 - 6*θ[0], 2*θ[1]])
    log_posterior = lambda θ, y=y, σ=σ: prior_log_dens(θ) - norm(y - F(θ))**2 / (2*σ**2) - 1*log(σ)
    grad_log_post = lambda θ, y=y, σ=σ: grad_log_prior(θ) + (y - F(θ))*grad_F(θ) / (σ**2)

    class BIPManifold(Manifold):
        def __init__(self, sigma, ystar):
            self.m = 1                 # One constraint F(theta, eta) = y
            self.d = 2                 # Dimension on manifold is 3 - 1 = 2.
            self.n = self.m + self.d   # Dimension of ambient space
            self.sigma = sigma
            self.ystar = ystar

        def q(self, xi):
            """Constraint for toy BIP."""
            return array([xi[1]**2 + 3 * xi[0]**2 * (xi[0]**2 - 1)]) + self.sigma*xi[2] - self.ystar

        def Q(self, xi):
            """Transpose of Jacobian for toy BIP. """
            return array([12*xi[0]**3 - 6*xi[0], 2*xi[1], self.sigma]).reshape(-1, self.m)

        def logpost(self, xi):
            """log posterior for c-rwm"""
            jac = self.Q(xi).T
            return - xi[:2]@xi[:2]/2 - xi[-1]**2/2 - np.log(jac@jac.T + self.sigma**2)[0, 0]/2

    for (i, (σ, ϵ)) in enumerate(product(σ_grid, ϵ_grid)):
        # Create initial points for C-HMC on the lifted manifold
        q_inits = [concatenate([θ, (y - forward_func(θ)) / σ]) for θ in θ_inits]    
        for qinit in q_inits:
            # Instantiate manifold
            manifold = BIPManifold(sigma=σ, ystar=y)
            logp = MVN(mean=zeros(manifold.d), cov=eye(manifold.d)).logpdf
            # Perform C-RWM 
            start_time = time.time()
            samples, nevals, accepted = zappa_sampling_storecomps(qinit, manifold, manifold.logpost, logp, n_sample, n_step*ϵ, tol=1e-8)
            runtime = time.time() - start_time
            crwm_av_miness_runtime.flat[i] += (min(ESS_univariate(samples)) / runtime) / len(q_inits)
            crwm_av_accept_probs.flat[i] += mean(accepted) / len(q_inits)
            crwm_av_njacob_evals.flat[i] += nevals['jacobian'] / len(q_inits)
    
    folder = "experiment39b/" #"dumper8/" #"experiment39/"

    save(folder + "CRWM_AP.npy", crwm_av_accept_probs)
    save(folder + "CRWM_NGRAD.npy", crwm_av_njacob_evals)
    save(folder + "CRWM_MINESS_OVER_RUNTIME.npy", crwm_av_miness_runtime)
    save(folder + "SIGMA_GRID.npy", σ_grid)   
    save(folder + "EPSILON_GRID.npy", ϵ_grid) 
