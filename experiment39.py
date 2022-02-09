# Experiment 39: The aim is to compare C-HMC with Hug/Thug on a Bayesian Inverse Problem.
# This is the toy Bayesian Inverse problem in Graham (although slightly different, taken from
# Graham's MICI notebook).
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

from mici.samplers import ChainState
from mici.systems import DenseConstrainedEuclideanMetricSystem as DCEMS
from mici.systems import EuclideanMetricSystem as EMS
from mici.systems import DenseRiemannianMetricSystem as DRMS
from mici.integrators import ConstrainedLeapfrogIntegrator as CLI
from mici.integrators import LeapfrogIntegrator as LI
from mici.integrators import ImplicitLeapfrogIntegrator as ILI
from mici.samplers import StaticMetropolisHMC as SMHMC
from mici.solvers import solve_projection_onto_manifold_newton

from Manifolds.GeneralizedEllipse import GeneralizedEllipse
from tangential_hug_functions import Hug, HugTangential, HugStepEJSD_Deterministic, HugTangentialStepEJSD_Deterministic
from utils import ESS_univariate

import time
import inspect
from functools import partial
from itertools import product
import symnum.numpy as snp
from symnum import (
    numpify, named_array, jacobian, grad, 
    vector_jacobian_product, matrix_hessian_product)
import numpy as np
import sympy
from autograd import grad as gradAD
import autograd.numpy as anp
import autograd.scipy as asp



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



if __name__ == "__main__":
    # Posterior Parameters
    σ = 0.1
    y = 1
    dim_θ = 2
    dim_y = 1

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
    hmc_av_accept_probs   = full((σ_grid.shape[0], ϵ_grid.shape[0]), nan)
    rmhmc_av_accept_probs = full((σ_grid.shape[0], ϵ_grid.shape[0]), nan)
    chmc_av_accept_probs  = full((σ_grid.shape[0], ϵ_grid.shape[0]), nan)
    hug_av_accept_probs   = full((σ_grid.shape[0], ϵ_grid.shape[0]), 0.0)
    thug_av_accept_probs  = full((σ_grid.shape[0], ϵ_grid.shape[0]), 0.0)
    hug_av_accept_probsL  = full((σ_grid.shape[0], ϵ_grid.shape[0]), 0.0)
    thug_av_accept_probsL = full((σ_grid.shape[0], ϵ_grid.shape[0]), 0.0)
    hmc_av_accept_probsL  = full((σ_grid.shape[0], ϵ_grid.shape[0]), 0.0)

    seed = 20200310
    rng = default_rng(seed)

    grad_neg_log_posterior_dens = grad(neg_log_posterior_dens)
    grad_and_val_neg_log_posterior_dens = grad(neg_log_posterior_dens, return_aux=True)
    vjp_metric = vector_jacobian_product(metric, return_aux=True)
    grad_neg_log_prior_dens = grad(neg_log_prior_dens)
    jacob_constr = jacobian(constr, return_aux=True)
    mhp_constr = matrix_hessian_product(constr, return_aux=True)



    _ = np.seterr(invalid='ignore', over='ignore')

    # Run HMC
    for (i, (σ, ϵ)) in enumerate(product(σ_grid, ϵ_grid)):
        system = EMS(neg_log_dens=partial(neg_log_posterior_dens, σ=σ, y=y), grad_neg_log_dens=partial(grad_and_val_neg_log_posterior_dens, σ=σ, y=y))
        integrator = LI(system, step_size=ϵ)
        sampler = SMHMC(system, integrator, rng, n_step=n_step)
        _, _, stats = sampler.sample_chains(n_sample, θ_inits, display_progress=False)
        hmc_av_accept_probs.flat[i] = concatenate([a for a in stats['accept_stat']]).mean()
    print("HMC finished.")

    # Run RM-HMC
    for (i, (σ, ϵ)) in enumerate(product(σ_grid, ϵ_grid)):
        system = DRMS(neg_log_dens=partial(neg_log_posterior_dens, σ=σ, y=y), grad_neg_log_dens=partial(grad_neg_log_posterior_dens, σ=σ, y=y), metric_func=partial(metric, σ=σ), vjp_metric_func=partial(vjp_metric, σ=σ))
        integrator = ILI(system, step_size=ϵ)
        sampler = SMHMC(system, integrator, rng, n_step=n_step // 2)
        _, _, stats = sampler.sample_chains(n_sample // 2, θ_inits, n_process=n_chain, display_progress=False)
        rmhmc_av_accept_probs.flat[i] = concatenate([a for a in stats['accept_stat']]).mean()
    print("RM-HMC finished.")

    # Run C-HMC
    for (i, (σ, ϵ)) in enumerate(product(σ_grid, ϵ_grid)):
        q_inits = [concatenate([θ, (y - forward_func(θ)) / σ]) for θ in θ_inits]
        system = DCEMS(neg_log_dens=neg_log_prior_dens, grad_neg_log_dens=grad_neg_log_prior_dens, dens_wrt_hausdorff=False, constr=partial(constr, σ=σ, y=y), jacob_constr=partial(jacob_constr, σ=σ, y=y), mhp_constr=partial(mhp_constr, σ=σ, y=y))
        integrator = CLI(system, step_size=ϵ, projection_solver=solve_projection_onto_manifold_newton)
        sampler = SMHMC(system, integrator, rng, n_step=n_step)
        _, _, stats = sampler.sample_chains(n_sample, q_inits, n_process=n_chain, display_progress=False)
        chmc_av_accept_probs.flat[i] = concatenate([a for a in stats['accept_stat']]).mean()
    print("C-HMC finished.")

    # Run Hug and Thug on the posterior distribution
    q = MVN(zeros(2), eye(2))
    prior_log_dens = lambda x: MVN(zeros(2), eye(2)).logpdf(x)
    grad_log_prior = lambda x: -x
    F = lambda θ: array([θ[1]**2 + 3 * θ[0]**2 * (θ[0]**2 - 1)])
    grad_F = lambda θ: array([12*θ[0]**3 - 6*θ[0], 2*θ[1]])
    log_posterior = lambda θ, y=y, σ=σ: prior_log_dens(θ) - norm(y - F(θ))**2 / (2*σ**2) - 1*log(σ)
    grad_log_post = lambda θ, y=y, σ=σ: grad_log_prior(θ) + (y - F(θ))*grad_F(θ) / (σ**2)
    #αs = array([0.95, 0.95, 0.95, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0])
    αs = array([0.99, 0.99, 0.99, 0.9, 0.9, 0.9, 0.9, 0.9, 0.7, 0.0, 0.0])
    for (i, (σ, ϵ)) in enumerate(product(σ_grid, ϵ_grid)):
        for θinit in θ_inits:
            logpi = lambda θ: log_posterior(θ, y=y, σ=σ)
            grad_logpi = lambda θ: grad_log_post(θ, y=y, σ=σ)
            # Perform Hug and Thug with the same velocities and random seeds
            x_hug  = θinit
            x_thug = θinit
            v0 = q.rvs(n_sample)
            logu = log(rand(n_sample))
            acc_hug = []
            acc_thug = []
            α = αs[σ_grid == σ][0] if ϵ == 1.0 else 0.0
            for iii in range(n_sample):
                x_hug, ahug, _, _, _  = HugStepEJSD_Deterministic(x_hug, v0[iii], logu[iii], ϵ * n_step, n_step,q, logpi, grad_logpi)
                x_thug, athug, _, _, _ = HugTangentialStepEJSD_Deterministic(x_thug, v0[iii], logu[iii], ϵ * n_step, n_step, α, q, logpi, grad_logpi)
                acc_hug.append(ahug)
                acc_thug.append(athug)
            hug_av_accept_probs.flat[i] += mean(acc_hug) / len(θ_inits)
            thug_av_accept_probs.flat[i] += mean(acc_thug) / len(θ_inits)
    print("Hug and Thug finished.")

    # Hug and Thug on the approximate lifted distribution (L) stands for lifted
    qL = MVN(zeros(3), eye(3))
    Gσ = lambda ξ, σ: F(ξ[:2]) + σ * ξ[-1]  # Function in ξ = [θ, η]
    logpriorL = lambda ξ: MVN(zeros(3), eye(3)).logpdf(ξ)
    def log_epanechnikov_kernel(ξ, ϵ, σ, y):
        u = norm(Gσ(ξ, σ) - y)
        with errstate(divide='ignore'):
            return log((3*(1 - (u**2 / (ϵ**2))) / (4*ϵ)) * float(norm(Gσ(ξ, σ) - y) <= ϵ))
    log_posteriorL = lambda ξ, ϵ, σ, y: logpriorL(ξ) + log_epanechnikov_kernel(ξ, ϵ, σ, y)
    grad_manifoldL = lambda ξ, σ: array([12*ξ[0]**3 - 6*ξ[0], 2*ξ[1], σ])
    for (i, (σ, ϵ)) in enumerate(product(σ_grid, ϵ_grid)):
        ξ_inits = [concatenate([θ, (y - forward_func(θ)) / σ]) for θ in θ_inits]
        for ξinit in ξ_inits:
            logpi = lambda ξ: log_posteriorL(ξ, ϵ=0.02, σ=σ, y=y)
            grad_logpi = lambda ξ: grad_manifoldL(ξ, σ=σ)
            # Perform Hug and Thug with the same velocities and random seeds
            x_hug  = ξinit
            x_thug = ξinit
            v0 = qL.rvs(n_sample)
            logu = log(rand(n_sample))
            acc_hug = []
            acc_thug = []
            if σ > 1e-2:
                α = 0.0
            elif σ <= 1e-2 and 1e-3 <= σ:
                α = 0.9
            else:
                α = 0.99
            for iii in range(n_sample):
                x_hug, ahug, _, _, _  = HugStepEJSD_Deterministic(x_hug, v0[iii], logu[iii], ϵ * n_step, n_step, qL, logpi, grad_logpi)
                x_thug, athug, _, _, _ = HugTangentialStepEJSD_Deterministic(x_thug, v0[iii], logu[iii], ϵ * n_step, n_step, α, qL, logpi, grad_logpi)
                acc_hug.append(ahug)
                acc_thug.append(athug)
            hug_av_accept_probsL.flat[i] += mean(acc_hug) / len(ξ_inits)
            thug_av_accept_probsL.flat[i] += mean(acc_thug) / len(ξ_inits)
    print("Hug and Thug on lifted distribution finished.")

    # Run HMC on the approximate lifted distribution (to compare to Hug and Thug)
    # logpriorL_AD = lambda ξ: -(3/2)*anp.log(2*anp.pi) - (ξ @ ξ) 
    # F_AD = lambda θ: anp.array([θ[1]**2 + 3 * θ[0]**2 * (θ[0]**2 - 1)])
    # Gσ_AD = lambda ξ, σ: F_AD(ξ[:2]) + σ * ξ[-1]
    # def log_epanechnikov_kernel_AD(ξ, ϵ, σ, y):
    #     u = abs(Gσ_AD(ξ, σ) - y)
    #     with errstate(divide='ignore'):
    #         return anp.log((3*(1 - (u**2 / (ϵ**2))) / (4*ϵ)) * float(abs(Gσ_AD(ξ, σ) - y) <= ϵ))
    # log_posteriorL_AD = lambda ξ, ϵ, σ, y: logpriorL_AD(ξ) + log_epanechnikov_kernel_AD(ξ, ϵ, σ, y)
    # for (i, (σ, ϵ)) in enumerate(product(σ_grid, ϵ_grid)):
    #     ξ_inits = [concatenate([θ, (y - forward_func(θ)) / σ]) for θ in θ_inits]
    #     def neglogpost(x):
    #         return -log_posteriorL_AD(x, 0.02, σ, y)
    #     system = EMS(neg_log_dens=neglogpost, grad_neg_log_dens=gradAD(neglogpost))
    #     integrator = LI(system, step_size=ϵ)
    #     sampler = SMHMC(system, integrator, rng, n_step=n_step)
    #     _, _, stats = sampler.sample_chains(n_sample, ξ_inits, display_progress=False)
    #     hmc_av_accept_probsL.flat[i] = concatenate([a for a in stats['accept_stat']]).mean()
    
    folder = "dumper8/" #"experiment39/"

    save(folder + "HMC_AP.npy", hmc_av_accept_probs)
    save(folder + "RMHMC_AP.npy", rmhmc_av_accept_probs)
    save(folder + "CHMC_AP.npy", chmc_av_accept_probs)
    save(folder + "HUG_AP.npy", hug_av_accept_probs)
    save(folder + "THUG_AP.npy", thug_av_accept_probs)
    save(folder + "HUGL_AP.npy", hug_av_accept_probsL)
    save(folder + "THUGL_AP.npy", thug_av_accept_probsL)
    save(folder + "HMC-L.npy", hmc_av_accept_probsL)

    save(folder + "ALPHAS.npy", αs)
    save(folder + "SIGMA_GRID.npy", σ_grid) 
    save(folder + "EPSILON_GRID.npy", ϵ_grid)
    
    
    




