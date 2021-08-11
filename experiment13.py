# Uniform Kernel. Uniform Prior. Use HOP to change energy level within the tube.
# Use expected mean squared jump distance
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save
from numpy.linalg import solve
from scipy.stats import multivariate_normal as MVN
from tangential_hug_functions import Hop, HugStepEJSD, HugTangentialStepEJSD
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


def runHH(x0, lam, N):
    """Runs HUG and HOP."""
    samples = x = x0
    accept1 = zeros(N)
    accept2 = zeros(N)
    esjd = 0.0
    esjd_grad = 0.0
    esjd_tan = 0.0
    for _ in range(N):
        y, a1, e, eg, et = HugStepEJSD(x, T, B, q, log_abc_posterior, grad_log_simulator)
        x, a2 = Hop(y, lam, kappa, log_abc_posterior, grad_log_simulator)
        samples = vstack((samples, y, x))
        accept1[_], accept2[_] = a1, a2
        esjd += e / N
        esjd_grad += eg / N 
        esjd_tan += et / N 
    return samples[1:], mean(accept1)*100, mean(accept2)*100, esjd, esjd_grad, esjd_tan


def runTH(x0, lam, N):
    """Runs HUG and HOP."""
    samples = x = x0
    accept1 = zeros(N)
    accept2 = zeros(N)
    esjd = 0.0
    esjd_grad = 0.0
    esjd_tan = 0.0
    for _ in range(N):
        y, a1, e, eg, et = HugTangentialStepEJSD(x, T, B, alpha, q, log_abc_posterior, grad_log_simulator)
        x, a2 = Hop(y, lam, kappa, log_abc_posterior, grad_log_simulator)
        samples = vstack((samples, y, x))
        accept1[_], accept2[_] = a1, a2
        esjd += e / N
        esjd_grad += eg / N 
        esjd_tan += et / N 
    return samples[1:], mean(accept1)*100, mean(accept2)*100, esjd, esjd_grad, esjd_tan
        

if __name__ == "__main__":
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
    kappa = 0.25
    n_runs = 10

    epsilons = [0.1, 0.001, 0.00001, 0.0000001]
    alphas = [0.1, 0.5, 0.9, 0.99, 0.999]

    # HUG
    THETA_ESS_HUG = zeros((n_runs, len(epsilons)))        # Univariate ESS for \theta chain
    U_ESS_HUG = zeros((n_runs, len(epsilons)))            # Univariate ESS for u chain
    ESS_LOGPI_HUG = zeros((n_runs, len(epsilons)))        # ESS for joint ABC posterior log p(\xi | y)
    ESS_JOINT_HUG = zeros((n_runs, len(epsilons)))        # multiESS on whole chain
    A_HUG = zeros((n_runs, len(epsilons)))                # Acceptance probability
    RMSE_HUG = zeros((n_runs, len(epsilons)))             # Root Mean Squared Error
    EJSD_HUG = zeros((n_runs, len(epsilons)))             # Full EJSD 
    G_EJSD_HUG = zeros((n_runs, len(epsilons)))           # EJSD for gradient only
    T_EJSD_HUG = zeros((n_runs, len(epsilons)))           # EJSD for tangent only
    A_HOP_HUG  = zeros((n_runs, len(epsilons)))           # Acceptance probability of HOP for HUG.
    N_UNIQUE_HUG = zeros((n_runs, len(epsilons)))         # Number of unique samples

    # THUG
    THETA_ESS_THUG = zeros((n_runs, len(epsilons), len(alphas)))      
    U_ESS_THUG = zeros((n_runs, len(epsilons), len(alphas)))  
    ESS_JOINT_THUG = zeros((n_runs, len(epsilons), len(alphas)))        
    ESS_LOGPI_THUG = zeros((n_runs, len(epsilons), len(alphas))) 
    A_THUG = zeros((n_runs, len(epsilons), len(alphas)))            
    RMSE_THUG = zeros((n_runs, len(epsilons), len(alphas)))    
    EJSD_THUG = zeros((n_runs, len(epsilons), len(alphas)))             
    G_EJSD_THUG = zeros((n_runs, len(epsilons), len(alphas)))           
    T_EJSD_THUG = zeros((n_runs, len(epsilons), len(alphas)))
    A_HOP_THUG = zeros((n_runs, len(epsilons), len(alphas)))    
    N_UNIQUE_THUG = zeros((n_runs, len(epsilons), len(alphas)))        


    for j, epsilon in enumerate(epsilons):
        lam = epsilon  
        for i in range(n_runs):
            # HUG
            hug, ahug, ahop, e, eg, et = runHH(x0, lam, N)
            THETA_ESS_HUG[i, j] = ESS_univariate(hug[:, 0])
            U_ESS_HUG[i, j] = ESS_univariate(hug[:, 1])
            ESS_JOINT_HUG[i, j] = ESS(hug)
            ESS_LOGPI_HUG[i, j] = ESS_univariate(log_abc_posterior_all(hug))
            A_HUG[i, j] = ahug
            A_HOP_HUG[i, j] = ahop
            RMSE_HUG[i, j] = sqrt(mean((target.logpdf(hug) - z0)**2))
            EJSD_HUG[i, j] = e
            G_EJSD_HUG[i, j] = eg
            T_EJSD_HUG[i, j] = et
            N_UNIQUE_HUG[i, j] = n_unique(hug)

            # THUG
            for k, alpha in enumerate(alphas):
                thug, athug, ahop, e, eg, et = runTH(x0, lam, N)
                THETA_ESS_THUG[i, j, k] = ESS_univariate(thug[:, 0])
                U_ESS_THUG[i, j, k] = ESS_univariate(thug[:, 1])
                ESS_JOINT_THUG[i, j, k] = ESS(thug)
                ESS_LOGPI_THUG[i, j, k] = ESS_univariate(log_abc_posterior_all(thug))
                A_THUG[i, j, k] = athug
                A_HOP_THUG[i, j, k] = ahop
                RMSE_THUG[i, j, k] = sqrt(mean((target.logpdf(thug) - z0)**2))
                EJSD_THUG[i, j, k] = e 
                G_EJSD_THUG[i, j, k] = eg 
                T_EJSD_THUG[i, j, k] = et
                N_UNIQUE_THUG[i, j, k] = n_unique(thug)


    save("experiment13/EPSILONS.npy", epsilons)
    save("experiment13/ALPHAS.npy", alphas)

    save("experiment13/THETA_ESS_HUG.npy", THETA_ESS_HUG)
    save("experiment13/U_ESS_HUG.npy", U_ESS_HUG)
    save("experiment13/ESS_JOINT_HUG.npy", ESS_JOINT_HUG)
    save("experiment13/ESS_LOGPI_HUG.npy", ESS_LOGPI_HUG)
    save("experiment13/A_HUG.npy", A_HUG)
    save("experiment13/RMSE_HUG.npy", RMSE_HUG)
    save("experiment13/EJSD_HUG.npy", EJSD_HUG)
    save("experiment13/G_EJSD_HUG.npy", G_EJSD_HUG)
    save("experiment13/T_EJSD_HUG.npy", T_EJSD_HUG)
    save("experiment13/A_HOP_HUG.npy", A_HOP_HUG)
    save("experiment13/N_UNIQUE_HUG.npy", N_UNIQUE_HUG)

    save("experiment13/THETA_ESS_THUG.npy", THETA_ESS_THUG)
    save("experiment13/U_ESS_THUG.npy", U_ESS_THUG)
    save("experiment13/ESS_JOINT_THUG.npy", ESS_JOINT_THUG)
    save("experiment13/ESS_LOGPI_THUG.npy", ESS_LOGPI_THUG)
    save("experiment13/A_THUG.npy", A_THUG)
    save("experiment13/RMSE_THUG.npy", RMSE_THUG)
    save("experiment13/EJSD_THUG.npy", EJSD_THUG)
    save("experiment13/G_EJSD_THUG.npy", G_EJSD_THUG)
    save("experiment13/T_EJSD_THUG.npy", T_EJSD_THUG)
    save("experiment13/A_HOP_THUG.npy", A_HOP_THUG)
    save("experiment13/N_UNIQUE_THUG.npy", N_UNIQUE_THUG)