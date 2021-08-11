# Experiment 17: HUG vs HUG-AR-rho (full AR process) in LFI setting without HOP.
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save
from numpy.linalg import solve
from scipy.stats import multivariate_normal as MVN
from tangential_hug_functions import Hug_EJSD, HugARrho_EJSD
from utils import ESS_univariate, ESS
from numpy.random import normal


def log_uniform_kernel(xi, epsilon):
    """Log density of uniform kernel. """
    with np.errstate(divide='ignore'):
        return log((abs(target.logpdf(xi) - z0) <= epsilon).astype('float64'))
    
def logprior_uniform(xi):
    """Log density for uniform prior p(xi) of parameters and latents U([-5,5]x[-5,5])."""
    with np.errstate(divide='ignore'):
        return log((abs(xi) <= 5.0).all().astype('float64'))

def logprior_uniform_all(xi):
    """Log density for uniform prior p(xi) of parameters and latents U([-5,5]x[-5,5])."""
    with np.errstate(divide='ignore'):
        return log((abs(xi) <= 5.0).all(axis=1).astype('float64'))
    
def log_abc_posterior(xi):
    """Log density of ABC posterior. Product of (param-latent) prior and uniform kernel."""
    return logprior_uniform(xi) + log_uniform_kernel(xi, epsilon)

def log_abc_posterior_all(xi):
    """Log density of ABC posterior. Product of (param-latent) prior and uniform kernel."""
    return logprior_uniform_all(xi) + log_uniform_kernel(xi, epsilon)
    
def grad_log_simulator(xi):
    """Gradient of log simulator N(mu, Sigma)."""
    return - solve(Sigma, xi)


# Target distribution is a diagonal MVN 
Sigma0 = diag([1.0, 5.0])
rho = 1.0
Sigma = rho * Sigma0
target = MVN(zeros(2), Sigma)

# Initial point on z0-contour
x0 = normal(size=2)                           # Keep initial point the same
z0 = target.logpdf(x0)                        # Feed through simulator

# Proposal for velocity in HUG/THUG
q = MVN(zeros(2), eye(2))

# Settings
T = 1.5
B = 5
N = 50000
n_runs = 10

epsilons = [0.1, 0.000001]
rhos = [0.5, 0.75, 0.95]
n_rhos = len(rhos)
n_epsilons = len(epsilons)

THETA_ESS_HUG    = zeros((n_runs, n_epsilons))
U_ESS_HUG        = zeros((n_runs, n_epsilons))
ESS_HUG          = zeros((n_runs, n_epsilons))
A_HUG            = zeros((n_runs, n_epsilons))
EJSD_HUG         = zeros((n_runs, n_epsilons))

THETA_ESS_HUG_AR = zeros((n_runs, n_epsilons, n_rhos))
U_ESS_HUG_AR     = zeros((n_runs, n_epsilons, n_rhos))
ESS_HUG_AR       = zeros((n_runs, n_epsilons, n_rhos))
A_HUG_AR         = zeros((n_runs, n_epsilons, n_rhos))
EJSD_HUG_AR      = zeros((n_runs, n_epsilons, n_rhos))

for i in range(n_runs):
    for j, epsilon in enumerate(epsilons):
        # Run standard THUG
        hug, ahug, ejsd = Hug_EJSD(x0, T, B, N, q, log_abc_posterior, grad_log_simulator)
        THETA_ESS_HUG[i, j] = ESS_univariate(hug[:, 0])
        U_ESS_HUG[i, j] = ESS_univariate(hug[:, 1])
        ESS_HUG[i, j] = ESS(hug)
        A_HUG[i, j] = mean(ahug) * 100
        EJSD_HUG[i, j] = mean(ejsd)

        for k, rho in enumerate(rhos):
            # Run THUG-AR
            hug_ar, ahug_ar, ejsd_ar = HugARrho_EJSD(x0, T, B, N, rho, q, log_abc_posterior, grad_log_simulator)
            THETA_ESS_HUG_AR[i, j, k] = ESS_univariate(hug_ar[:, 0])
            U_ESS_HUG_AR[i, j, k] = ESS_univariate(hug_ar[:, 1])
            ESS_HUG_AR[i, j, k] = ESS(hug_ar)
            A_HUG_AR[i, j, k] = mean(ahug_ar) * 100 
            EJSD_HUG_AR[i, j, k] = mean(ejsd_ar)


folder = "experiment17/"

save(folder + "RHOS.npy", rhos)
save(folder + "EPSILONS.npy", epsilons)

save(folder + "THETA_ESS_HUG.npy", THETA_ESS_HUG)
save(folder + "U_ESS_HUG.npy", U_ESS_HUG)
save(folder + "ESS_HUG.npy", ESS_HUG)
save(folder + "A_HUG.npy", A_HUG)
save(folder + "EJSD_HUG.npy", EJSD_HUG)

save(folder + "THETA_ESS_HUG_AR.npy", THETA_ESS_HUG_AR)
save(folder + "U_ESS_HUG_AR.npy", U_ESS_HUG_AR)
save(folder + "ESS_HUG_AR.npy", ESS_HUG_AR)
save(folder + "A_HUG_AR.npy", A_HUG_AR)
save(folder + "EJSD_HUG_AR.npy", EJSD_HUG_AR)