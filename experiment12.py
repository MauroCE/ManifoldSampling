# HUG and THUG in likelihood-free setting with GAUSSIAN KERNEL and Gaussian prior p(\xi).
# This does not uses a large Hessian.
import numpy as np
from numpy import zeros, diag, eye, log, pi, sqrt, vstack, exp, mean, save
from numpy.linalg import solve
from scipy.stats import multivariate_normal as MVN
from tangential_hug_functions import Hug, HugTangential
from utils import ESS, ESS_univariate, quick_MVN_scatter, prep_contour
import matplotlib.pyplot as plt
from scipy.stats import norm as ndist
from numpy.random import uniform, normal
import tensorflow_probability as tfp

def log_normal_kernel(xi, epsilon):
    """Log density of normal kernel. """
    return MVN(z0, epsilon**2).logpdf(target.logpdf(xi))
    
def logprior_normal(xi):
    """Log density for normal prior p(xi) of parameters and latents N(0, I)."""
    return MVN(zeros(2), eye(2)).logpdf(xi)
    
def log_abc_posterior(xi, eps):
    """Log density of ABC posterior. Product of (param-latent) prior and uniform kernel."""
    return logprior_normal(xi) + log_normal_kernel(xi, eps)
    
def grad_log_simulator(xi):
    """Gradient of log simulator N(mu, Sigma)."""
    return - solve(Sigma, xi)


# Target distribution is a diagonal MVN
Sigma0 = diag([1.0, 5.0])
rho = 1.0
Sigma = rho * Sigma0
target = MVN(zeros(2), Sigma)

# Initial point on z0-contour
x0 = normal(size=2)        # Sample parameter and latent from the prior
z0 = target.logpdf(x0)     # Feed through simulator

# Proposal for velocity in HUG/THUG
q = MVN(zeros(2), eye(2))


# Settings
T = 1.0
B = 5
N = 2000
n_runs = 12

epsilons = [0.1, 0.001, 0.00001]
alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

# HUG
uESS_HUG = zeros((n_runs, len(epsilons)))        # Univariate ESS for \theta chain
ESS_LOGPI_HUG = zeros((n_runs, len(epsilons)))   # ESS for joint ABC posterior log p(\xi | y)
A_HUG = zeros((n_runs, len(epsilons)))           # Acceptance probability
RMSE_HUG = zeros((n_runs, len(epsilons)))        # Root Mean Squared Error

# THUG
uESS_THUG = zeros((n_runs, len(epsilons), len(alphas)))  
ESS_LOGPI_THUG = zeros((n_runs, len(epsilons), len(alphas))) 
A_THUG = zeros((n_runs, len(epsilons), len(alphas)))            
RMSE_THUG = zeros((n_runs, len(epsilons), len(alphas)))    


for j, epsilon in enumerate(epsilons):
    for i in range(n_runs):
        # Posterior
        logpi = lambda xi: log_abc_posterior(xi, epsilon)
        # HUG
        hug, ahug   = Hug(x0, T, B, N, q, logpi, grad_log_simulator)
        uESS_HUG[i, j] = ESS_univariate(hug[:, 0])
        ESS_LOGPI_HUG[i, j] = ESS_univariate(logpi(hug))
        A_HUG[i, j] = mean(ahug) * 100
        RMSE_HUG[i, j] = sqrt(mean((target.logpdf(hug) - z0)**2))

        # THUG
        for k, alpha in enumerate(alphas):
            thug, athug = HugTangential(x0, T, B, N, alpha, q, logpi, grad_log_simulator)
            uESS_THUG[i, j, k] = ESS_univariate(thug[:, 0])
            ESS_LOGPI_THUG[i, j, k] = ESS_univariate(logpi(thug))
            A_THUG[i, j, k] = mean(athug) * 100 
            RMSE_THUG[i, j, k] = sqrt(mean((target.logpdf(thug) - z0)**2))


save("experiment12/EPSILONS.npy", epsilons)
save("experiment12/ALPHAS.npy", alphas)

save("experiment12/ESS_HUG.npy", uESS_HUG)
save("experiment12/ESS_LOGPI_HUG.npy", ESS_LOGPI_HUG)
save("experiment12/A_HUG.npy", A_HUG)
save("experiment12/RMSE_HUG.npy", RMSE_HUG)

save("experiment12/ESS_THUG.npy", uESS_THUG)
save("experiment12/ESS_LOGPI_THUG.npy", ESS_LOGPI_THUG)
save("experiment12/A_THUG.npy", A_THUG)
save("experiment12/RMSE_THUG.npy", RMSE_THUG)