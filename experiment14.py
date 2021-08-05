# Experiment 14: The aim is to determine whether THUG with AR process is better than THUG in sampling 
# from a manifold. In this case we don't use any hop, only THUG/THUG-AR. The AR process is a "degenerate" 
# one: with probability `prob` we either keep exactly the same SPHERICAL velocity, otherwise we refresh 
# it completely.
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save
from numpy.linalg import solve
from scipy.stats import multivariate_normal as MVN
from tangential_hug_functions import HugTangential, HugTangentialAR
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
N = 1000
n_runs = 10

epsilons = [0.1, 0.0001, 0.000001]
alpha = 0.9
probs = [0.25, 0.5, 0.75, 0.95]
n_probs = len(probs)
n_epsilons = len(epsilons)

THETA_ESS_THUG    = zeros((n_runs, n_epsilons))
U_ESS_THUG        = zeros((n_runs, n_epsilons))
ESS_THUG          = zeros((n_runs, n_epsilons))
A_THUG            = zeros((n_runs, n_epsilons))

THETA_ESS_THUG_AR = zeros((n_runs, n_epsilons, n_probs))
U_ESS_THUG_AR     = zeros((n_runs, n_epsilons, n_probs))
ESS_THUG_AR       = zeros((n_runs, n_epsilons, n_probs))
A_THUG_AR         = zeros((n_runs, n_epsilons, n_probs))

for i in range(n_runs):
    for j, epsilon in enumerate(epsilons):
        # Run standard THUG
        thug, athug = HugTangential(x0, T, B, N, alpha, q, log_abc_posterior, grad_log_simulator)
        THETA_ESS_THUG[i, j] = ESS_univariate(thug[:, 0])
        U_ESS_THUG[i, j] = ESS_univariate(thug[:, 1])
        ESS_THUG[i, j] = ESS(thug)
        A_THUG[i, j] = mean(athug) * 100

        for k, prob in enumerate(probs):
            # Run THUG-AR
            thug_ar, athug_ar = HugTangentialAR(x0, T, B, N, alpha, prob, q, log_abc_posterior, grad_log_simulator)
            THETA_ESS_THUG_AR[i, j, k] = ESS_univariate(thug_ar[:, 0])
            U_ESS_THUG_AR[i, j, k] = ESS_univariate(thug_ar[:, 1])
            ESS_THUG_AR[i, j, k] = ESS(thug_ar)
            A_THUG_AR[i, j, k] = mean(athug_ar) * 100 


folder = "experiment14/"

save(folder + "PROBS.npy", probs)
save(folder + "EPSILONS.npy", epsilons)

save(folder + "THETA_ESS_THUG.npy", THETA_ESS_THUG)
save(folder + "U_ESS_THUG.npy", U_ESS_THUG)
save(folder + "ESS_THUG.npy", ESS_THUG)
save(folder + "A_THUG.npy", A_THUG)

save(folder + "THETA_ESS_THUG_AR.npy", THETA_ESS_THUG_AR)
save(folder + "U_ESS_THUG_AR.npy", U_ESS_THUG_AR)
save(folder + "ESS_THUG_AR.npy", ESS_THUG_AR)
save(folder + "A_THUG_AR.npy", A_THUG_AR)