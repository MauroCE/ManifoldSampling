# Experiment 19: HUG + HOP vs HUG-AR-rho + HOP in LFI setting with uniform kernel and uniform prior.
# We use the full AR process
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save
from numpy.linalg import solve
from scipy.stats import multivariate_normal as MVN
from tangential_hug_functions import Hop, HugStepEJSD, HugARStepEJSD  # Notice this works for full AR too
from utils import ESS_univariate, ESS, n_unique
from numpy.random import normal, uniform


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


def runHH_ARrho(x0, lam, N, q, rho):
    """Runs HUG-AR-rho and HOP."""
    assert abs(rho) <=1, "Rho must be between -1 and 1"
    samples = x = x0
    accept1 = zeros(N)
    accept2 = zeros(N)
    esjd = 0.0
    esjd_grad = 0.0
    esjd_tan = 0.0
    v = q.rvs()
    for _ in range(N):
        y, v, a1, e, eg, et = HugARStepEJSD(x, v, T, B, q, log_abc_posterior, grad_log_simulator)
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

    epsilons = [0.1, 0.0001, 0.0000001]
    rhos = [0.5, 0.75, 0.95]

    n_epsilons = len(epsilons)
    n_rhos = len(rhos)

    # HUG
    THETA_ESS_HUG = zeros((n_runs, n_epsilons))        # Univariate ESS for \theta chain
    U_ESS_HUG = zeros((n_runs, n_epsilons))            # Univariate ESS for u chain
    ESS_HUG = zeros((n_runs, n_epsilons))        # multiESS on whole chain
    A_HUG = zeros((n_runs, n_epsilons))                # Acceptance probability
    RMSE_HUG = zeros((n_runs, n_epsilons))             # Root Mean Squared Error
    EJSD_HUG = zeros((n_runs, n_epsilons))             # Full EJSD 
    G_EJSD_HUG = zeros((n_runs, n_epsilons))           # EJSD for gradient only
    T_EJSD_HUG = zeros((n_runs, n_epsilons))           # EJSD for tangent only
    A_HOP_HUG  = zeros((n_runs, n_epsilons))           # Acceptance probability of HOP for HUG.
    N_UNIQUE_HUG = zeros((n_runs, n_epsilons))         # Number of unique samples

    # HUG_AR
    THETA_ESS_HUG_AR = zeros((n_runs, n_epsilons, n_rhos))      
    U_ESS_HUG_AR = zeros((n_runs, n_epsilons, n_rhos))  
    ESS_HUG_AR = zeros((n_runs, n_epsilons, n_rhos))        
    A_HUG_AR = zeros((n_runs, n_epsilons, n_rhos))            
    RMSE_HUG_AR = zeros((n_runs, n_epsilons, n_rhos))    
    EJSD_HUG_AR = zeros((n_runs, n_epsilons, n_rhos))             
    G_EJSD_HUG_AR = zeros((n_runs, n_epsilons, n_rhos))           
    T_EJSD_HUG_AR = zeros((n_runs, n_epsilons, n_rhos))
    A_HOP_HUG_AR = zeros((n_runs, n_epsilons, n_rhos))    
    N_UNIQUE_HUG_AR = zeros((n_runs, n_epsilons, n_rhos))        


    for j, epsilon in enumerate(epsilons):
        lam = epsilon  
        for i in range(n_runs):
            # HUG
            hug, ahug, ahop, e, eg, et = runHH(x0, lam, N)
            THETA_ESS_HUG[i, j] = ESS_univariate(hug[:, 0])
            U_ESS_HUG[i, j] = ESS_univariate(hug[:, 1])
            ESS_HUG[i, j] = ESS(hug)
            A_HUG[i, j] = ahug
            A_HOP_HUG[i, j] = ahop
            RMSE_HUG[i, j] = sqrt(mean((target.logpdf(hug) - z0)**2))
            EJSD_HUG[i, j] = e
            G_EJSD_HUG[i, j] = eg
            T_EJSD_HUG[i, j] = et
            N_UNIQUE_HUG[i, j] = n_unique(hug)

            # HUG-AR
            for k, rho in enumerate(rhos):
                hug_ar, ahug_ar, ahop, e, eg, et = runHH_ARrho(x0, lam, N, q, rho)
                THETA_ESS_HUG_AR[i, j, k] = ESS_univariate(hug_ar[:, 0])
                U_ESS_HUG_AR[i, j, k] = ESS_univariate(hug_ar[:, 1])
                ESS_HUG_AR[i, j, k] = ESS(hug_ar)
                A_HUG_AR[i, j, k] = ahug_ar
                A_HOP_HUG_AR[i, j, k] = ahop
                RMSE_HUG_AR[i, j, k] = sqrt(mean((target.logpdf(hug_ar) - z0)**2))
                EJSD_HUG_AR[i, j, k] = e 
                G_EJSD_HUG_AR[i, j, k] = eg 
                T_EJSD_HUG_AR[i, j, k] = et
                N_UNIQUE_HUG_AR[i, j, k] = n_unique(hug_ar)

    
    folder = "experiment19/"

    save(folder + "EPSILONS.npy", epsilons)
    save(folder + "RHOS.npy", rhos)

    save(folder + "THETA_ESS_HUG.npy", THETA_ESS_HUG)
    save(folder + "U_ESS_HUG.npy", U_ESS_HUG)
    save(folder + "ESS_HUG.npy", ESS_HUG)
    save(folder + "A_HUG.npy", A_HUG)
    save(folder + "RMSE_HUG.npy", RMSE_HUG)
    save(folder + "EJSD_HUG.npy", EJSD_HUG)
    save(folder + "G_EJSD_HUG.npy", G_EJSD_HUG)
    save(folder + "T_EJSD_HUG.npy", T_EJSD_HUG)
    save(folder + "A_HOP_HUG.npy", A_HOP_HUG)
    save(folder + "N_UNIQUE_HUG.npy", N_UNIQUE_HUG)

    save(folder + "THETA_ESS_HUG_AR.npy", THETA_ESS_HUG_AR)
    save(folder + "U_ESS_HUG_AR.npy", U_ESS_HUG_AR)
    save(folder + "ESS_HUG_AR.npy", ESS_HUG_AR)
    save(folder + "A_HUG_AR.npy", A_HUG_AR)
    save(folder + "RMSE_HUG_AR.npy", RMSE_HUG_AR)
    save(folder + "EJSD_HUG_AR.npy", EJSD_HUG_AR)
    save(folder + "G_EJSD_HUG_AR.npy", G_EJSD_HUG_AR)
    save(folder + "T_EJSD_HUG_AR.npy", T_EJSD_HUG_AR)
    save(folder + "A_HOP_HUG_AR.npy", A_HOP_HUG_AR)
    save(folder + "N_UNIQUE_HUG_AR.npy", N_UNIQUE_HUG_AR)