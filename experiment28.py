# Experiment 28: HUG+HOP vs HUG+HOP+AR on 2D Gaussian example.
from re import S
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save, exp, linspace, pi
from numpy.linalg import solve
from scipy.stats import multivariate_normal as MVN
from tangential_hug_functions import Hop_Deterministic, HugStepEJSD_Deterministic, HugStepEJSD_DeterministicAR
from utils import ESS_univariate, ESS, n_unique
from numpy.random import normal, rand, uniform
from statsmodels.tsa.stattools import acf
from Manifolds.RotatedEllipse import RotatedEllipse
import time


def log_uniform_kernel(xi, epsilon):
    """Log density of uniform kernel. """
    with np.errstate(divide='ignore'):
        return log((abs(target.logpdf(xi) - z0) <= epsilon).astype('float64'))
    
def logprior_uniform(xi):
    """Log density for uniform prior p(xi) of parameters and latents U([-5,5]x[-5,5])."""
    with np.errstate(divide='ignore'):
        return log((abs(xi) <= 50.0).all().astype('float64'))

def logprior_uniform_all(xi):
    """Log density for uniform prior p(xi) of parameters and latents U([-5,5]x[-5,5])."""
    with np.errstate(divide='ignore'):
        return log((abs(xi) <= 50.0).all(axis=1).astype('float64'))
    
def log_abc_posterior(xi):
    """Log density of ABC posterior. Product of (param-latent) prior and uniform kernel."""
    return logprior_uniform(xi) + log_uniform_kernel(xi, epsilon)

def log_abc_posterior_all(xi):
    """Log density of ABC posterior. Product of (param-latent) prior and uniform kernel."""
    return logprior_uniform_all(xi) + log_uniform_kernel(xi, epsilon)
    
def grad_log_simulator(xi):
    """Gradient of log simulator N(mu, Sigma)."""
    return - solve(Sigma, xi)


def experiment(x00, T, N, nlags, rho = 0.95):
    """Runs Hug+Hop and Hug+Hop+AR using the same velocities and the same random seeds."""
    ### COMMON VARIABLES
    v = q.rvs(N)
    log_uniforms1 = log(rand(N))     # Log uniforms for the HUG kernels
    log_uniforms2 = log(rand(N))     # Log uniforms for the HOP kernel
    u = MVN(zeros(2), eye(2)).rvs(N) # Original velocities for HOP kernel
    ### STORAGE (HUG + HOP)
    hh = x00              # Initial sample
    ahh1 = 0.0       # Acceptance probability for HUG kernel
    ahh2 = 0.0       # Acceptance probability for HOP kernel (when used with HUG)
    ehh = 0.0             # EJSD
    eghh = 0.0            # EJSD in Gradient direction
    ethh = 0.0            # EJSD in Tangent direction
    ### STORAGE (HUG + HOP with AR process
    ar = x00 
    aar1 = 0.0
    aar2 = 0.0
    ear  = 0.0
    egar = 0.0
    etar = 0.0
    ### HUG + HOP
    x = x00
    for i in range(N):
        y, a1, e, eg, et = HugStepEJSD_Deterministic(x, v[i], log_uniforms1[i], T, B, q, log_abc_posterior, grad_log_simulator)
        x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, log_abc_posterior, grad_log_simulator)
        hh = vstack((hh, y, x))
        ahh1 += a1 * 100 / N
        ahh2 += a2 * 100 / N
        ehh += e / N
        eghh += eg / N 
        ethh += et / N 
    # COMPUTE ESS AND OTHER METRICS FOR HUG
    hh = hh[1:]
    hh_esst = ESS_univariate(hh[::2, 0])     # ESS for theta
    hh_essu = ESS_univariate(hh[::2, 1])     # ESS for u
    hh_essj = ESS(hh[::2])                   # ESS joint
    hh_rmse = sqrt(mean((target.logpdf(hh) - z0)**2))  # RMSE on energy
    hh_uniq = n_unique(hh)                             # Number of unique samples
    hh_act  = acf(hh[::2, 0], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for theta (remove the first 1.0)
    hh_acu  = acf(hh[::2, 1], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for u
    ### HUG + HOP + AR process
    x = x00
    rho_val = 1.0
    velocity = v[0]
    for i in range(N):
        velocity = rho_val * velocity + np.sqrt(1 - rho_val**2) * v[i]
        y, velocity, a1, e, eg, et = HugStepEJSD_DeterministicAR(x, velocity, log_uniforms1[i], T, B, q, log_abc_posterior, grad_log_simulator)
        x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, log_abc_posterior, grad_log_simulator)
        ar = vstack((ar, y, x))
        aar1 += a1 * 100 / N
        aar2 += a2 * 100 / N
        ear += e / N
        egar += eg / N 
        etar += et / N 
        rho_val = rho
    # COMPUTE ESS AND OTHER METRICS FOR HUG
    ar = ar[1:]
    ar_esst = ESS_univariate(ar[::2, 0])     # ESS for theta
    ar_essu = ESS_univariate(ar[::2, 1])     # ESS for u
    ar_essj = ESS(ar[::2])                   # ESS joint
    ar_rmse = sqrt(mean((target.logpdf(ar) - z0)**2))  # RMSE on energy
    ar_uniq = n_unique(ar)                             # Number of unique samples
    ar_act  = acf(ar[::2, 0], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for theta (remove the first 1.0)
    ar_acu  = acf(ar[::2, 1], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for u

    # RETURN EVERYTHING
    out = {
        'HH': {
            'A1': ahh1,
            'A2': ahh2,
            'E': ehh,
            'EG': eghh, 
            'ET': ethh,
            'ESS_T': hh_esst,
            'ESS_U': hh_essu,
            'ESS_J': hh_essj,
            'RMSE': hh_rmse,
            'UNIQUE': hh_uniq,
            'AC_T': hh_act,
            'AC_U': hh_acu
        },
        'AR': {
            'A1': aar1,
            'A2': aar2,
            'E': ear,
            'EG': egar, 
            'ET': etar, 
            'ESS_T': ar_esst,
            'ESS_U': ar_essu,
            'ESS_J': ar_essj,
            'RMSE': ar_rmse,
            'UNIQUE': ar_uniq,
            'AC_T': ar_act,
            'AC_U': ar_acu
        }
    }
    return out

        

if __name__ == "__main__":
    # Target distribution is a diagonal MVN
    Sigma = diag([1.0, 5.0])  # theta, u
    target = MVN(zeros(2), Sigma)

    # Proposal for velocity in HUG/THUG
    q = MVN(zeros(2), eye(2))

    # Sample a point at the beginning just to obtain some energy level
    z0 = -2.9513586307684885 #target.logpdf(target.rvs())
    # Function to sample a new point on the contour
    new_point = lambda: RotatedEllipse(zeros(2), Sigma, exp(z0)).to_cartesian(uniform(0, 2*pi))

    # Settings
    B = 5 
    N = 50000
    kappa = 0.25    
    n_runs = 20 #15
    nlags = 20
    rho  = 0.7

    Ts = [10, 1, 0.1, 0.01] #[7, 5, 3, 1, 0.1, 0.01]
    epsilons = [0.1, 0.001, 0.00001, 0.0000001]
    n_epsilons = len(epsilons)
    n_T = len(Ts)

    # HUG
    THETA_ESS_HUG = zeros((n_runs, n_epsilons, n_T))        # Univariate ESS for \theta chain
    U_ESS_HUG = zeros((n_runs, n_epsilons, n_T))            # Univariate ESS for u chain
    ESS_JOINT_HUG = zeros((n_runs, n_epsilons, n_T))        # multiESS on whole chain
    A_HUG = zeros((n_runs, n_epsilons, n_T))                # Acceptance probability
    RMSE_HUG = zeros((n_runs, n_epsilons, n_T))             # Root Mean Squared Error
    EJSD_HUG = zeros((n_runs, n_epsilons, n_T))             # Full EJSD 
    G_EJSD_HUG = zeros((n_runs, n_epsilons, n_T))           # EJSD for gradient only
    T_EJSD_HUG = zeros((n_runs, n_epsilons, n_T))           # EJSD for tangent only
    A_HOP_HUG  = zeros((n_runs, n_epsilons, n_T))           # Acceptance probability of HOP for HUG.
    N_UNIQUE_HUG = zeros((n_runs, n_epsilons, n_T))         # Number of unique samples
    THETA_AC_HUG = zeros((n_runs, n_epsilons, n_T, nlags))  # Autocorrelation for theta
    U_AC_HUG = zeros((n_runs, n_epsilons, n_T, nlags))      # Autocorrelation for u

    # THUG
    THETA_ESS_AR = zeros((n_runs, n_epsilons, n_T))      
    U_ESS_AR = zeros((n_runs, n_epsilons, n_T))  
    ESS_JOINT_AR = zeros((n_runs, n_epsilons, n_T))        
    A_AR = zeros((n_runs, n_epsilons, n_T))            
    RMSE_AR = zeros((n_runs, n_epsilons, n_T))    
    EJSD_AR = zeros((n_runs, n_epsilons, n_T))             
    G_EJSD_AR = zeros((n_runs, n_epsilons, n_T))           
    T_EJSD_AR = zeros((n_runs, n_epsilons, n_T))
    A_HOP_AR = zeros((n_runs, n_epsilons, n_T))    
    N_UNIQUE_AR = zeros((n_runs, n_epsilons, n_T))     
    THETA_AC_AR = zeros((n_runs, n_epsilons, n_T, nlags)) 
    U_AC_AR = zeros((n_runs, n_epsilons, n_T, nlags)) 

    initial_time = time.time()
    for i in range(n_runs):
        # We need a new point for each run, but then must be the same for all other settings
        initial_point = new_point()
        for j, epsilon in enumerate(epsilons):
            lam = epsilon # For HOP
            for k, T in enumerate(Ts):
                out = experiment(initial_point, T, N, nlags, rho)
                # Store HUG results
                THETA_ESS_HUG[i, j, k]   = out['HH']['ESS_T']
                U_ESS_HUG[i, j, k]       = out['HH']['ESS_U']
                ESS_JOINT_HUG[i, j, k]   = out['HH']['ESS_J']
                A_HUG[i, j, k]           = out['HH']['A1']
                A_HOP_HUG[i, j, k]       = out['HH']['A2']
                RMSE_HUG[i, j, k]        = out['HH']['RMSE']
                EJSD_HUG[i, j, k]        = out['HH']['E']
                G_EJSD_HUG[i, j, k]      = out['HH']['EG']
                T_EJSD_HUG[i, j, k]      = out['HH']['ET']
                N_UNIQUE_HUG[i, j, k]    = out['HH']['UNIQUE']
                THETA_AC_HUG[i, j, k, :] = out['HH']['AC_T']
                U_AC_HUG[i, j, k, :]     = out['HH']['AC_U']
                # Store THUG results
                THETA_ESS_AR[i, j, k]   = out['AR']['ESS_T']
                U_ESS_AR[i, j, k]       = out['AR']['ESS_U']
                ESS_JOINT_AR[i, j, k]   = out['AR']['ESS_J']
                A_AR[i, j, k]           = out['AR']['A1']
                A_HOP_AR[i, j, k]       = out['AR']['A2']
                RMSE_AR[i, j, k]        = out['AR']['RMSE']
                EJSD_AR[i, j, k]        = out['AR']['E']
                G_EJSD_AR[i, j, k]      = out['AR']['EG']
                T_EJSD_AR[i, j, k]      = out['AR']['ET']
                N_UNIQUE_AR[i, j, k]    = out['AR']['UNIQUE']
                THETA_AC_AR[i, j, k, :] = out['AR']['AC_T']
                U_AC_AR[i, j, k, :]     = out['AR']['AC_U']

    print("Total time: ", time.time() - initial_time)

    # Save results
    folder = "dumper/"

    save(folder + "EPSILONS.npy", epsilons)
    save(folder + "TS.npy", Ts)
    save(folder + "TIME.npy", np.array([time.time() - initial_time]))
    save(folder + "RHO.npy", rho)
    save(folder + "N.npy", N)
    save(folder + "N_RUNS.npy", n_runs)

    save(folder + "THETA_ESS_HUG.npy", THETA_ESS_HUG)
    save(folder + "U_ESS_HUG.npy", U_ESS_HUG)
    save(folder + "ESS_JOINT_HUG.npy", ESS_JOINT_HUG)
    save(folder + "A_HUG.npy", A_HUG)
    save(folder + "RMSE_HUG.npy", RMSE_HUG)
    save(folder + "EJSD_HUG.npy", EJSD_HUG)
    save(folder + "G_EJSD_HUG.npy", G_EJSD_HUG)
    save(folder + "T_EJSD_HUG.npy", T_EJSD_HUG)
    save(folder + "A_HOP_HUG.npy", A_HOP_HUG)
    save(folder + "N_UNIQUE_HUG.npy", N_UNIQUE_HUG)
    save(folder + "THETA_AC_HUG.npy", THETA_AC_HUG)
    save(folder + "U_AC_HUG.npy", U_AC_HUG)

    save(folder + "THETA_ESS_AR.npy", THETA_ESS_AR)
    save(folder + "U_ESS_AR.npy", U_ESS_AR)
    save(folder + "ESS_JOINT_AR.npy", ESS_JOINT_AR)
    save(folder + "A_AR.npy", A_AR)
    save(folder + "RMSE_AR.npy", RMSE_AR)
    save(folder + "EJSD_AR.npy", EJSD_AR)
    save(folder + "G_EJSD_AR.npy", G_EJSD_AR)
    save(folder + "T_EJSD_AR.npy", T_EJSD_AR)
    save(folder + "A_HOP_AR.npy", A_HOP_AR)
    save(folder + "N_UNIQUE_AR.npy", N_UNIQUE_AR)
    save(folder + "THETA_AC_AR.npy", THETA_AC_AR)
    save(folder + "U_AC_AR.npy", U_AC_AR)

