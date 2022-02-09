# Experiment 38: The aim is to compare C-HMC with Hug/Thug on an ABC-like example.
# The manifold is a contour of a mixture of Gaussians. The prior is N(0, 1) and the]
# kernel is Epanochnikov.
import numpy as np
from numpy.linalg import norm, solve, det
from numpy.random import default_rng, rand, randn
from numpy import log, exp, zeros, eye, pi, array, diag, sqrt, cumsum, where, vstack, array
from numpy import apply_along_axis, outer, errstate, save, mean, sqrt
from scipy.stats import multivariate_normal as MVN
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time
from warnings import catch_warnings, filterwarnings

from mici.samplers import ChainState
from mici.systems import DenseConstrainedEuclideanMetricSystem as DCEMS
from mici.integrators import ConstrainedLeapfrogIntegrator as CLI
from mici.samplers import StaticMetropolisHMC as SMHMC

from Manifolds.GeneralizedEllipse import GeneralizedEllipse
from tangential_hug_functions import Hug, HugTangential
from utils import ESS_univariate


if __name__ == "__main__":
    # Settings for the components of the Mixture of Gaussians
    σxs  = sqrt([3.0, 1.0, 0.05, 0.5, 2.0, 0.05])
    σys  = sqrt([3.0, 1.0, 0.05, 0.5, 2.0, 0.05])
    ρs   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cs   = [1 / len(σxs)] * len(σxs)
    μs   = [[0, 0], [2, 0], [0, 2], [-2, 0], [0, -2], [1, 1]]
    Σs   = [array([[σx**2, ρ*σx*σy], [ρ*σx*σy, σy**2]]) for (σx, σy, ρ) in zip(σxs, σys, ρs)]
    MVNs = [MVN(μ, Σ) for (μ, Σ) in zip(μs, Σs)]

    # Deterministic Function and Gradient
    f          = lambda xi: sum([c * MVN.pdf(xi) for (c, MVN) in zip(cs, MVNs)])
    grad_f     = lambda xi: sum([-c* MVN(μ, Σ).pdf(xi) * solve(Σ, xi - μ) for (c, μ, Σ) in zip(cs, μs, Σs)])
    sampleMG   = lambda: MVNs[where(rand() > cumsum(np.r_[0., cs]))[0][-1] - 1].rvs()

    # Root Mean Squared Error
    rmse = lambda x: sqrt(mean(x**2))

    # Decide which contour will be the manifold
    z0 = 0.007

    # Function to find initial point on manifold
    def new_point():
        with catch_warnings():
            filterwarnings('error')
            found = False
            while not found:
                try:
                    guess = randn(2)  # Construct guess
                    point = fsolve(lambda x: array([f(x) - z0, 0.]), guess, fprime=lambda x: vstack((grad_f(x), zeros(2))))
                    found = True
                    return point
                except Warning:
                    continue

    # Settings
    N_CHMC = 10000  # Number of samples for C-HMC
    δ_CHMC = 0.2   # Step-size for C-HMC (0.2)
    n_CHMC = 5     # Number of integrator steps to simulate in each transition 
    T_CHMC = δ_CHMC * n_CHMC

    ϵ_HUG = 0.00001
    N_HUG = N_CHMC
    T_HUG = T_CHMC
    B_HUG = n_CHMC

    ϵ_THUG = ϵ_HUG
    N_THUG = N_HUG
    T_THUG = T_HUG
    B_THUG = B_HUG

    EPSILONS = [0.001, 0.0000001, 0.0000000001] #[0.001, 0.00001, 0.0000001]
    N_EPSILON = len(EPSILONS)

    N_RUNS = 10
    ALPHAS = [0.9, 0.99, 0.999]
    if len(ALPHAS) != len(EPSILONS):
        raise ValueError("The number of alphas should be the same as epsilons")

    #### functions for CHMC
    logprior      = lambda xi: MVN(zeros(2), eye(2)).logpdf(xi)
    grad_logprior = lambda xi: -xi

    constr        = lambda q: array([f(q) - z0])
    jacob_constr  = lambda q: grad_f(q).reshape(1, -1)
    neg_log_dens  = lambda q: -log(norm(grad_f(q))) + logprior(q)

    def grad_neg_log_dens(x):
        n = len(μs)
        Cmatrix = outer(cs, cs)
        denom = 0.0
        numer = zeros(2)
        for i in range(n):
            for j in range(n):
                val = solve(Σs[i].T, solve(Σs[j], x - μs[j]))
                denom += 2 * Cmatrix[i, j] * (x - μs[i]) @ val
                numer += 2 * Cmatrix[i, j] * val
        return - grad_logprior(x) + (numer / denom)

    def trace_func(state):
        x, y = state.pos
        return {'x': x, 'y': y}

    #### Functions for HUG
    q = MVN(zeros(2), eye(2))

    def log_epanechnikov_kernel(xi, epsilon, ystar):
        u = norm(f(xi) - ystar)
        with errstate(divide='ignore'):
            return log((3*(1 - (u**2 / (epsilon**2))) / (4*epsilon)) * float(u <= epsilon))

    ### SET UP STORAGE FOR CHMC
    N_CONSTR_EVAL_CHMC   = zeros(N_RUNS)
    N_JAC_CONSTR_CHMC    = zeros(N_RUNS)
    N_GRAD_LOG_DENS_CHMC = zeros(N_RUNS)
    N_GRAM_CHMC          = zeros(N_RUNS)
    N_LOG_DENS_CHMC      = zeros(N_RUNS)
    LOGPI_ESS_CHMC       = zeros(N_RUNS)
    CONSTR_PER_ESS_CHMC  = zeros(N_RUNS)
    C_AND_J_PER_ESS_CHMC = zeros(N_RUNS)
    C_J_GLD_PER_ESS_CHMC = zeros(N_RUNS)
    ACCEPT_STAT_CHMC     = zeros(N_RUNS)
    RMSE_CHMC            = zeros(N_RUNS)

    ### SET UP STORAGE FOR HUG
    N_CONSTR_EVAL_HUG   = zeros((N_RUNS, N_EPSILON))
    N_JAC_CONSTR_HUG    = zeros((N_RUNS, N_EPSILON))
    LOGPI_ESS_HUG       = zeros((N_RUNS, N_EPSILON))
    CONSTR_PER_ESS_HUG  = zeros((N_RUNS, N_EPSILON))
    C_AND_J_PER_ESS_HUG = zeros((N_RUNS, N_EPSILON))
    ACCEPT_STAT_HUG     = zeros((N_RUNS, N_EPSILON))
    RMSE_HUG            = zeros((N_RUNS, N_EPSILON))

    ### SET UP STORAGE FOR THUG
    N_CONSTR_EVAL_THUG   = zeros((N_RUNS, N_EPSILON))
    N_JAC_CONSTR_THUG    = zeros((N_RUNS, N_EPSILON))
    LOGPI_ESS_THUG       = zeros((N_RUNS, N_EPSILON))
    CONSTR_PER_ESS_THUG  = zeros((N_RUNS, N_EPSILON))
    C_AND_J_PER_ESS_THUG = zeros((N_RUNS, N_EPSILON))
    ACCEPT_STAT_THUG     = zeros((N_RUNS, N_EPSILON))
    RMSE_THUG            = zeros((N_RUNS, N_EPSILON))

    initial_time = time.time()

    INITIAL_POINTS = []

    for i in range(N_RUNS):
        # Initial point on manifold
        xi0 = new_point()
        INITIAL_POINTS.append(xi0)

        ### C-HMC
        system     = DCEMS(neg_log_dens, constr, jacob_constr=jacob_constr, grad_neg_log_dens=grad_neg_log_dens)
        integrator = CLI(system, step_size=δ_CHMC)
        sampler    = SMHMC(system, integrator, default_rng(), n_step=n_CHMC)
        init_state_CHMC = ChainState(pos=xi0, mom=None, dir=1, _call_counts={})  
        final_state, trace, stat = sampler.sample_chain(n_iter=N_CHMC, init_state=init_state_CHMC, trace_funcs=[trace_func], display_progress=False)

        ### COMPUTATIONS FOR C-HMC
        CHMC_stats              = {key[0].split('.', 1)[1]: value for (key, value) in final_state._call_counts.items()}
        N_CONSTR_EVAL_CHMC[i]   = CHMC_stats['constr']
        N_JAC_CONSTR_CHMC[i]    = CHMC_stats['jacob_constr']
        N_GRAD_LOG_DENS_CHMC[i] = CHMC_stats['grad_neg_log_dens']
        N_GRAM_CHMC[i]          = CHMC_stats['gram']
        N_LOG_DENS_CHMC[i]      = CHMC_stats['neg_log_dens']
        samples_CHMC            = vstack((trace['x'], trace['y'])).T
        dens_values_CHMC        = exp(-apply_along_axis(neg_log_dens, 1, samples_CHMC))
        LOGPI_ESS_CHMC[i]       = ESS_univariate(dens_values_CHMC)
        CONSTR_PER_ESS_CHMC[i]  = N_CONSTR_EVAL_CHMC[i] / LOGPI_ESS_CHMC[i]
        C_AND_J_PER_ESS_CHMC[i] = (N_CONSTR_EVAL_CHMC[i] + N_JAC_CONSTR_CHMC[i]) / LOGPI_ESS_CHMC[i]
        C_J_GLD_PER_ESS_CHMC[i] = (N_CONSTR_EVAL_CHMC[i] + N_JAC_CONSTR_CHMC[i] + N_GRAD_LOG_DENS_CHMC[i]) / LOGPI_ESS_CHMC[i]
        ACCEPT_STAT_CHMC[i]     = stat['accept_stat'].mean()
        RMSE_CHMC[i]            = rmse(apply_along_axis(lambda xi: f(xi), 1, samples_CHMC) - z0)

        ### RUN HUG AND THUG ONE TIME FOR EACH EPSILON
        for j in range(N_EPSILON):
            # Functions for HUG
            logpi_HUG = lambda xi: logprior(xi) + log_epanechnikov_kernel(xi, EPSILONS[j], z0)
            grad_HUG  = lambda xi: grad_f(xi)
            
            # Functions for THUG
            logpi_THUG = lambda xi: logprior(xi) + log_epanechnikov_kernel(xi, EPSILONS[j], z0)
            grad_THUG  = lambda xi: grad_f(xi)
            
            # SAMPLE HUG AND THUG
            hug_samples, acceptance_hug = Hug(xi0, T_HUG, B_HUG, N_HUG, q, logpi_HUG, grad_HUG)
            thug_samples, acceptance_thug = HugTangential(xi0, T_THUG, B_THUG, N_THUG, ALPHAS[j], q, logpi_THUG, grad_THUG)
            
            ### COMPUTATIONS FOR HUG
            dens_values_HUG           = exp(apply_along_axis(logpi_HUG, 1, hug_samples))
            LOGPI_ESS_HUG[i, j]       = ESS_univariate(dens_values_HUG)
            N_CONSTR_EVAL_HUG[i, j]   = N_HUG + 1
            N_JAC_CONSTR_HUG[i, j]    = B_HUG * N_HUG
            CONSTR_PER_ESS_HUG[i, j]  = N_CONSTR_EVAL_HUG[i, j] / LOGPI_ESS_HUG[i, j]
            C_AND_J_PER_ESS_HUG[i, j] = (N_CONSTR_EVAL_HUG[i, j] + N_JAC_CONSTR_HUG[i, j]) / LOGPI_ESS_HUG[i, j]
            ACCEPT_STAT_HUG[i, j]     = acceptance_hug.mean()
            RMSE_HUG[i, j]            = rmse(apply_along_axis(lambda xi: f(xi), 1, hug_samples) - z0)

            ### COMPUTATIONS FOR THUG
            dens_values_THUG           = exp(apply_along_axis(logpi_THUG, 1, thug_samples))
            LOGPI_ESS_THUG[i, j]       = ESS_univariate(dens_values_THUG)
            N_CONSTR_EVAL_THUG[i, j]   = N_THUG + 1
            N_JAC_CONSTR_THUG[i, j]    = 1 + (B_THUG + 1) * N_THUG
            CONSTR_PER_ESS_THUG[i, j]  = N_CONSTR_EVAL_THUG[i, j] / LOGPI_ESS_THUG[i, j]
            C_AND_J_PER_ESS_THUG[i, j] = (N_CONSTR_EVAL_THUG[i, j] + N_JAC_CONSTR_THUG[i, j]) / LOGPI_ESS_THUG[i, j]
            ACCEPT_STAT_THUG[i, j]     = acceptance_thug.mean()
            RMSE_THUG[i, j]            = rmse(apply_along_axis(lambda xi: f(xi), 1, thug_samples) - z0)

    print("Total time: ", time.time() - initial_time)

    ### SAVE ONTO FOLDER
    folder = "dumper7/" #"experiment38/"

    save(folder + "N_CONSTR_EVAL_CHMC.npy", N_CONSTR_EVAL_CHMC)
    save(folder + "N_JAC_CONSTR_CHMC.npy", N_JAC_CONSTR_CHMC)
    save(folder + "N_GRAD_LOG_DENS_CHMC.npy", N_GRAD_LOG_DENS_CHMC)
    save(folder + "N_GRAM_CHMC.npy", N_GRAM_CHMC)
    save(folder + "N_LOG_DENS_CHMC.npy", N_LOG_DENS_CHMC)
    save(folder + "LOGPI_ESS_CHMC.npy", LOGPI_ESS_CHMC)
    save(folder + "CONSTR_PER_ESS_CHMC.npy", CONSTR_PER_ESS_CHMC)
    save(folder + "C_AND_J_PER_ESS_CHMC.npy", C_AND_J_PER_ESS_CHMC)
    save(folder + "C_J_GLD_PER_ESS_CHMC.npy", C_J_GLD_PER_ESS_CHMC)
    save(folder + "ACCEPT_STAT_CHMC.npy", ACCEPT_STAT_CHMC)
    save(folder + "RMSE_CHMC.npy", RMSE_CHMC)

    save(folder + "N_CONSTR_EVAL_HUG.npy", N_CONSTR_EVAL_HUG)
    save(folder + "N_JAC_CONSTR_HUG.npy", N_JAC_CONSTR_HUG)
    save(folder + "LOGPI_ESS_HUG.npy", LOGPI_ESS_HUG)
    save(folder + "CONSTR_PER_ESS_HUG.npy", CONSTR_PER_ESS_HUG)
    save(folder + "C_AND_J_PER_ESS_HUG.npy", C_AND_J_PER_ESS_HUG)
    save(folder + "ACCEPT_STAT_HUG.npy", ACCEPT_STAT_HUG)
    save(folder + "RMSE_HUG.npy", RMSE_HUG)

    save(folder + "N_CONSTR_EVAL_THUG.npy", N_CONSTR_EVAL_THUG)
    save(folder + "N_JAC_CONSTR_THUG.npy", N_JAC_CONSTR_THUG)
    save(folder + "LOGPI_ESS_THUG.npy", LOGPI_ESS_THUG)
    save(folder + "CONSTR_PER_ESS_THUG.npy", CONSTR_PER_ESS_THUG)
    save(folder + "C_AND_J_PER_ESS_THUG.npy", C_AND_J_PER_ESS_THUG)
    save(folder + "ACCEPT_STAT_THUG.npy", ACCEPT_STAT_THUG)
    save(folder + "RMSE_THUG.npy", RMSE_THUG)

    save(folder + "EPSILONS.npy", EPSILONS)
    save(folder + "ALPHAS.npy", ALPHAS)
    save(folder + "INITIAL_POINTS.npy", vstack(INITIAL_POINTS))

