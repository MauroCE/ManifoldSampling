# Experiment 24: Uniform Kernel. Uniform Prior. HUG+HOP vs THUG+HOP when dimensionality increases.
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save, exp, linspace, pi
from numpy.linalg import solve
from scipy.stats import multivariate_normal as MVN
from tangential_hug_functions import HugTangentialStepEJSD_Deterministic, Hop_Deterministic, HugStepEJSD_Deterministic
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


def experiment(x00, T, N, alphas, nlags):
    """Runs Hug+Hop and THUG+HOP using the same velocities and the same random seeds.
    We also try to limit the noise in the HOP kernel by sampling the u variables beforehand.
    I run THUG for all values of alpha with the randomness fixed. 
    This is 1 run, for 1 epsilon. It does 1 HUG+HOP and then THUG+HOP for all alphas.
    T1: T for HUG
    T2: T for THUG
    """
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
    ### STORAGE (THUG + HOP) I MUST STORE FOR ALL ALPHAS
    ath1 = zeros(n_alphas)
    ath2 = zeros(n_alphas)
    eth  = zeros(n_alphas)
    egth = zeros(n_alphas)
    etth = zeros(n_alphas)
    ### ADDITIONAL STORAGE FOR THUG
    th_esst = zeros(n_alphas)
    th_essu = zeros(n_alphas)
    th_essj = zeros(n_alphas)
    th_rmse = zeros(n_alphas)
    th_uniq = zeros(n_alphas)
    th_act  = zeros((n_alphas, nlags))
    th_acu  = zeros((n_alphas, nlags))
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
    ### THUG + HOP
    for k, alpha in enumerate(alphas):
        x = x00
        th = x00      # RESTART THE SAMPLES FROM SCRATCH
        for i in range(N):
            y, a1, e, eg, et = HugTangentialStepEJSD_Deterministic(x, v[i], log_uniforms1[i], T, B, alpha, q, log_abc_posterior, grad_log_simulator)
            x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, log_abc_posterior, grad_log_simulator)
            th = vstack((th, y, x))
            ath1[k] += a1 * 100 / N
            ath2[k] += a2 * 100 / N
            eth[k]  += e / N
            egth[k] += eg / N 
            etth[k] += et / N 
        ### COMPUTE ESS AND OTHER METRISC FOR THUG
        th = th[1:]
        th_esst[k] = ESS_univariate(th[::2, 0])     # ESS for theta
        th_essu[k] = ESS_univariate(th[::2, 1])     # ESS for u
        th_essj[k] = ESS(th[::2])                   # ESS joint
        th_rmse[k] = sqrt(mean((target.logpdf(th) - z0)**2))  # RMSE on energy
        th_uniq[k] = n_unique(th)                             # Number of unique samples
        th_act[k] = acf(th[::2, 0], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for theta
        th_acu[k] = acf(th[::2, 1], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for u
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
        'TH': {
            'A1': ath1,
            'A2': ath2,
            'E': eth,
            'EG': egth, 
            'ET': etth, 
            'ESS_T': th_esst,
            'ESS_U': th_essu,
            'ESS_J': th_essj,
            'RMSE': th_rmse,
            'UNIQUE': th_uniq,
            'AC_T': th_act,
            'AC_U': th_acu
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
    n_runs = 50 #15
    nlags = 20

    Ts = [10, 1, 0.1, 0.01] #[7, 5, 3, 1, 0.1, 0.01]
    epsilons = [0.1, 0.001, 0.00001, 0.0000001]
    alphas = [0.5, 0.9, 0.99, 0.999]
    n_epsilons = len(epsilons)
    n_alphas = len(alphas)
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
    THETA_ESS_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))      
    U_ESS_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))  
    ESS_JOINT_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))        
    A_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))            
    RMSE_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))    
    EJSD_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))             
    G_EJSD_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))           
    T_EJSD_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))
    A_HOP_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))    
    N_UNIQUE_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))     
    THETA_AC_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas, nlags)) 
    U_AC_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas, nlags)) 

    initial_time = time.time()
    for i in range(n_runs):
        print("Run: ", i)
        # We need a new point for each run, but then must be the same for all other settings
        initial_point = new_point()
        for j, epsilon in enumerate(epsilons):
            lam = epsilon # For HOP
            for k, T in enumerate(Ts):
                out = experiment(initial_point, T, N, alphas, nlags)
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
                THETA_ESS_THUG[i, j, k, :]   = out['TH']['ESS_T']
                U_ESS_THUG[i, j, k, :]       = out['TH']['ESS_U']
                ESS_JOINT_THUG[i, j, k, :]   = out['TH']['ESS_J']
                A_THUG[i, j, k, :]           = out['TH']['A1']
                A_HOP_THUG[i, j, k, :]       = out['TH']['A2']
                RMSE_THUG[i, j, k, :]        = out['TH']['RMSE']
                EJSD_THUG[i, j, k, :]        = out['TH']['E'] 
                G_EJSD_THUG[i, j, k, :]      = out['TH']['EG'] 
                T_EJSD_THUG[i, j, k, :]      = out['TH']['ET']
                N_UNIQUE_THUG[i, j, k, :]    = out['TH']['UNIQUE']
                THETA_AC_THUG[i, j, k, :, :] = out['TH']['AC_T']
                U_AC_THUG[i, j, k, :, :]     = out['TH']['AC_U']

    # Save results
    folder = "experiment13full4/"

    save(folder + "EPSILONS.npy", epsilons)
    save(folder + "ALPHAS.npy", alphas)
    save(folder + "TS.npy", Ts)
    save(folder + "TIME.npy", np.array([time.time() - initial_time]))

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

    save(folder + "THETA_ESS_THUG.npy", THETA_ESS_THUG)
    save(folder + "U_ESS_THUG.npy", U_ESS_THUG)
    save(folder + "ESS_JOINT_THUG.npy", ESS_JOINT_THUG)
    save(folder + "A_THUG.npy", A_THUG)
    save(folder + "RMSE_THUG.npy", RMSE_THUG)
    save(folder + "EJSD_THUG.npy", EJSD_THUG)
    save(folder + "G_EJSD_THUG.npy", G_EJSD_THUG)
    save(folder + "T_EJSD_THUG.npy", T_EJSD_THUG)
    save(folder + "A_HOP_THUG.npy", A_HOP_THUG)
    save(folder + "N_UNIQUE_THUG.npy", N_UNIQUE_THUG)
    save(folder + "THETA_AC_THUG.npy", THETA_AC_THUG)
    save(folder + "U_AC_THUG.npy", U_AC_THUG)

