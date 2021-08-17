# Experiment 21: THUG-AR-rho + HOP vs THUG+HOP in LFI setting with uniform kernel and uniform prior.
# We use the full AR process. We focus on alpha=0.99 and on a small epsilon.
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save
from numpy.linalg import solve
from scipy.stats import multivariate_normal as MVN
from tangential_hug_functions import Hop, HugTangentialStepEJSD, HugTangentialStepEJSD_AR  # Notice this works for full AR too
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


def runTH(x0, lam, N, alpha):
    """Runs THUG and HOP."""
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
        

def runTH_ARrho(x0, lam, N, q, rho, alpha):
    """Runs THUG-AR-rho and HOP."""
    assert abs(rho) <=1, "Rho must be between -1 and 1"
    samples = x = x0
    accept1 = zeros(N)
    accept2 = zeros(N)
    esjd = 0.0
    esjd_grad = 0.0
    esjd_tan = 0.0
    v = q.rvs()
    for _ in range(N):
        y, v, a1, e, eg, et = HugTangentialStepEJSD_AR(x, v, T, B, alpha, q, log_abc_posterior, grad_log_simulator)
        x, a2 = Hop(y, lam, kappa, log_abc_posterior, grad_log_simulator)
        samples = vstack((samples, y, x))
        accept1[_], accept2[_] = a1, a2
        esjd += e / N
        esjd_grad += eg / N 
        esjd_tan += et / N 
        v = rho*v + np.sqrt(1 - rho**2)*q.rvs()
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

    # Proposal for velocity in HUG/HUG_AR
    q = MVN(zeros(2), eye(2))


    # Settings
    T = 1.5
    B = 5
    N = 50000
    kappa = 0.25
    n_runs = 10
    alpha = 0.99

    epsilons = [0.0000001]
    rhos = [0.5, 0.75, 0.95]

    n_epsilons = len(epsilons)
    n_rhos = len(rhos)

    # THUG
    THETA_ESS_THUG = zeros((n_runs, n_epsilons))        # Univariate ESS for \theta chain
    U_ESS_THUG = zeros((n_runs, n_epsilons))            # Univariate ESS for u chain
    ESS_THUG = zeros((n_runs, n_epsilons))        # multiESS on whole chain
    A_THUG = zeros((n_runs, n_epsilons))                # Acceptance probability
    RMSE_THUG = zeros((n_runs, n_epsilons))             # Root Mean Squared Error
    EJSD_THUG = zeros((n_runs, n_epsilons))             # Full EJSD 
    G_EJSD_THUG = zeros((n_runs, n_epsilons))           # EJSD for gradient only
    T_EJSD_THUG = zeros((n_runs, n_epsilons))           # EJSD for tangent only
    A_HOP_THUG  = zeros((n_runs, n_epsilons))           # Acceptance probability of HOP for THUG.
    N_UNIQUE_THUG = zeros((n_runs, n_epsilons))         # Number of unique samples

    # THUG_AR
    THETA_ESS_THUG_AR = zeros((n_runs, n_epsilons, n_rhos))      
    U_ESS_THUG_AR = zeros((n_runs, n_epsilons, n_rhos))  
    ESS_THUG_AR = zeros((n_runs, n_epsilons, n_rhos))        
    A_THUG_AR = zeros((n_runs, n_epsilons, n_rhos))            
    RMSE_THUG_AR = zeros((n_runs, n_epsilons, n_rhos))    
    EJSD_THUG_AR = zeros((n_runs, n_epsilons, n_rhos))             
    G_EJSD_THUG_AR = zeros((n_runs, n_epsilons, n_rhos))           
    T_EJSD_THUG_AR = zeros((n_runs, n_epsilons, n_rhos))
    A_HOP_THUG_AR = zeros((n_runs, n_epsilons, n_rhos))    
    N_UNIQUE_THUG_AR = zeros((n_runs, n_epsilons, n_rhos))        


    for j, epsilon in enumerate(epsilons):
        lam = epsilon  
        for i in range(n_runs):
            # THUG + HOP
            thug, athug, ahop, e, eg, et = runTH(x0, lam, N, alpha)
            THETA_ESS_THUG[i, j] = ESS_univariate(thug[:, 0])
            U_ESS_THUG[i, j] = ESS_univariate(thug[:, 1])
            ESS_THUG[i, j] = ESS(thug)
            A_THUG[i, j] = athug
            A_HOP_THUG[i, j] = ahop
            RMSE_THUG[i, j] = sqrt(mean((target.logpdf(thug) - z0)**2))
            EJSD_THUG[i, j] = e
            G_EJSD_THUG[i, j] = eg
            T_EJSD_THUG[i, j] = et
            N_UNIQUE_THUG[i, j] = n_unique(thug)

            # THUG-AR
            for k, rho in enumerate(rhos):
                thug_ar, athug_ar, ahop, e, eg, et = runTH_ARrho(x0, lam, N, q, rho, alpha)
                THETA_ESS_THUG_AR[i, j, k] = ESS_univariate(thug_ar[:, 0])
                U_ESS_THUG_AR[i, j, k] = ESS_univariate(thug_ar[:, 1])
                ESS_THUG_AR[i, j, k] = ESS(thug_ar)
                A_THUG_AR[i, j, k] = athug_ar
                A_HOP_THUG_AR[i, j, k] = ahop
                RMSE_THUG_AR[i, j, k] = sqrt(mean((target.logpdf(thug_ar) - z0)**2))
                EJSD_THUG_AR[i, j, k] = e 
                G_EJSD_THUG_AR[i, j, k] = eg 
                T_EJSD_THUG_AR[i, j, k] = et
                N_UNIQUE_THUG_AR[i, j, k] = n_unique(thug_ar)

    
    folder = "experiment21/"

    save(folder + "EPSILONS.npy", epsilons)
    save(folder + "RHOS.npy", rhos)

    save(folder + "THETA_ESS_THUG.npy", THETA_ESS_THUG)
    save(folder + "U_ESS_THUG.npy", U_ESS_THUG)
    save(folder + "ESS_THUG.npy", ESS_THUG)
    save(folder + "A_THUG.npy", A_THUG)
    save(folder + "RMSE_THUG.npy", RMSE_THUG)
    save(folder + "EJSD_THUG.npy", EJSD_THUG)
    save(folder + "G_EJSD_THUG.npy", G_EJSD_THUG)
    save(folder + "T_EJSD_THUG.npy", T_EJSD_THUG)
    save(folder + "A_HOP_THUG.npy", A_HOP_THUG)
    save(folder + "N_UNIQUE_THUG.npy", N_UNIQUE_THUG)

    save(folder + "THETA_ESS_THUG_AR.npy", THETA_ESS_THUG_AR)
    save(folder + "U_ESS_THUG_AR.npy", U_ESS_THUG_AR)
    save(folder + "ESS_THUG_AR.npy", ESS_THUG_AR)
    save(folder + "A_THUG_AR.npy", A_THUG_AR)
    save(folder + "RMSE_THUG_AR.npy", RMSE_THUG_AR)
    save(folder + "EJSD_THUG_AR.npy", EJSD_THUG_AR)
    save(folder + "G_EJSD_THUG_AR.npy", G_EJSD_THUG_AR)
    save(folder + "T_EJSD_THUG_AR.npy", T_EJSD_THUG_AR)
    save(folder + "A_HOP_THUG_AR.npy", A_HOP_THUG_AR)
    save(folder + "N_UNIQUE_THUG_AR.npy", N_UNIQUE_THUG_AR)