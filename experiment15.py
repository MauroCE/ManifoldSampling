# Experiment 15: The aim is to determine whether THUG with AR process is better than THUG in sampling 
# from a manifold. In this case we don't use any hop, only THUG/THUG-AR. The AR process is the full AR process
#     v0 * rho + sqrt((1 - rho**2)) * w_0
# rho represents the correlation between two successive velocities
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save
from numpy.linalg import solve
from scipy.stats import multivariate_normal as MVN
from tangential_hug_functions import HugTangential_EJSD, HugTangentialARrho_EJSD
from utils import ESS_univariate, ESS, n_unique
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
alpha = 0.99
rhos = [0.25, 0.5, 0.75, 0.95]
n_rhos = len(rhos)
n_epsilons = len(epsilons)

THETA_ESS_THUG    = zeros((n_runs, n_epsilons))
U_ESS_THUG        = zeros((n_runs, n_epsilons))
ESS_THUG          = zeros((n_runs, n_epsilons))
A_THUG            = zeros((n_runs, n_epsilons))
EJSD_THUG         = zeros((n_runs, n_epsilons))
N_UNIQUE_THUG     = zeros((n_runs, n_epsilons))

THETA_ESS_THUG_AR = zeros((n_runs, n_epsilons, n_rhos))
U_ESS_THUG_AR     = zeros((n_runs, n_epsilons, n_rhos))
ESS_THUG_AR       = zeros((n_runs, n_epsilons, n_rhos))
A_THUG_AR         = zeros((n_runs, n_epsilons, n_rhos))
EJSD_THUG_AR      = zeros((n_runs, n_epsilons, n_rhos))
N_UNIQUE_THUG_AR  = zeros((n_runs, n_epsilons, n_rhos))

for i in range(n_runs):
    for j, epsilon in enumerate(epsilons):
        # Run standard THUG
        thug, athug, ejsd = HugTangential_EJSD(x0, T, B, N, alpha, q, log_abc_posterior, grad_log_simulator)
        THETA_ESS_THUG[i, j] = ESS_univariate(thug[:, 0])
        U_ESS_THUG[i, j] = ESS_univariate(thug[:, 1])
        ESS_THUG[i, j] = ESS(thug)
        A_THUG[i, j] = mean(athug) * 100
        EJSD_THUG[i, j] = mean(ejsd)
        N_UNIQUE_THUG[i, j] = n_unique(thug)

        for k, rho in enumerate(rhos):
            # Run THUG-AR
            thug_ar, athug_ar, ejsd_ar = HugTangentialARrho_EJSD(x0, T, B, N, alpha, rho, q, log_abc_posterior, grad_log_simulator)
            THETA_ESS_THUG_AR[i, j, k] = ESS_univariate(thug_ar[:, 0])
            U_ESS_THUG_AR[i, j, k] = ESS_univariate(thug_ar[:, 1])
            ESS_THUG_AR[i, j, k] = ESS(thug_ar)
            A_THUG_AR[i, j, k] = mean(athug_ar) * 100 
            EJSD_THUG_AR[i, j, k] = ejsd_ar
            N_UNIQUE_THUG_AR[i, j, k] = n_unique(thug_ar)


folder = "experiment15/"

save(folder + "RHOS.npy", rhos)
save(folder + "EPSILONS.npy", epsilons)
save(folder + "ALPHA.npy", alpha)

save(folder + "THETA_ESS_THUG.npy", THETA_ESS_THUG)
save(folder + "U_ESS_THUG.npy", U_ESS_THUG)
save(folder + "ESS_THUG.npy", ESS_THUG)
save(folder + "A_THUG.npy", A_THUG)
save(folder + "EJSD_THUG.npy", EJSD_THUG)
save(folder + "N_UNIQUE_THUG.npy", N_UNIQUE_THUG)

save(folder + "THETA_ESS_THUG_AR.npy", THETA_ESS_THUG_AR)
save(folder + "U_ESS_THUG_AR.npy", U_ESS_THUG_AR)
save(folder + "ESS_THUG_AR.npy", ESS_THUG_AR)
save(folder + "A_THUG_AR.npy", A_THUG_AR)
save(folder + "EJSD_THUG_AR.npy", EJSD_THUG_AR)
save(folder + "N_UNIQUE_THUG_AR.npy", N_UNIQUE_THUG_AR)