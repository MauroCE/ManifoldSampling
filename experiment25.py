# Experiment 25: Similar to experiment 23 but using AR process
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save, exp, linspace, pi
from numpy.linalg import solve
from scipy.stats import multivariate_normal as MVN
from tangential_hug_functions import HugRotatedStepEJSD_AR_Deterministic
from tangential_hug_functions import Hop_Deterministic
from tangential_hug_functions import HugStepEJSD_Deterministic
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


def experiment(x00, T, N, alphas, nlags, rho=0.9):
    """Runs Hug+Hop and RotatedHUG+HOP using the same velocities and the same random seeds.
    We also try to limit the noise in the HOP kernel by sampling the u variables beforehand.
    I run RotatedHUG for all values of alpha with the randomness fixed. 
    This is 1 run, for 1 epsilon. It does 1 HUG+HOP and then THUG+HOP for all alphas.
    T1: T for HUG
    T2: T for RotatedHUG
    """
    ### COMMON VARIABLES
    v = q.rvs(N)
    log_uniforms1 = log(rand(N))     # Log uniforms for the HUG kernels
    log_uniforms2 = log(rand(N))     # Log uniforms for the HOP kernel
    u = MVN(zeros(2), eye(2)).rvs(N) # Original velocities for HOP kernel
    ### STORAGE (HUG + HOP)
    hh = x00              # Initial sample
    ahh1 = 0.0            # Acceptance probability for HUG kernel
    ahh2 = 0.0            # Acceptance probability for HOP kernel (when used with HUG)
    ehh = 0.0             # EJSD
    eghh = 0.0            # EJSD in Gradient direction
    ethh = 0.0            # EJSD in Tangent direction
    ### STORAGE (RotatedHUG + HOP) I MUST STORE FOR ALL ALPHAS
    arh1 = zeros(n_alphas)
    arh2 = zeros(n_alphas)
    erh  = zeros(n_alphas)
    egrh = zeros(n_alphas)
    etrh = zeros(n_alphas)
    ### ADDITIONAL STORAGE FOR RotatedHUG
    rh_esst = zeros(n_alphas)
    rh_essu = zeros(n_alphas)
    rh_essj = zeros(n_alphas)
    rh_rmse = zeros(n_alphas)
    rh_uniq = zeros(n_alphas)
    rh_act  = zeros((n_alphas, nlags))
    rh_acu  = zeros((n_alphas, nlags))
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
    ### RotatedHUG + HOP
    for k, alpha in enumerate(alphas):
        x = x00
        rh = x00      # RESTART THE SAMPLES FROM SCRATCH
        velocity = v[0]
        rho_value = 1.0    # will be changed to rho after 1st iteration
        for i in range(N):
            velocity = rho_value*velocity + sqrt(1 - rho_value**2)*v[i]
            y, velocity, a1, e, eg, et = HugRotatedStepEJSD_AR_Deterministic(x, velocity, log_uniforms1[i], T, B, alpha, q, log_abc_posterior, grad_log_simulator)
            x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, log_abc_posterior, grad_log_simulator)
            rh = vstack((rh, y, x))
            arh1[k] += a1 * 100 / N
            arh2[k] += a2 * 100 / N
            erh[k]  += e / N
            egrh[k] += eg / N 
            etrh[k] += et / N 
            rho_value = rho
        ### COMPUTE ESS AND OTHER METRISC FOR RotatedHUG
        rh = rh[1:]
        rh_esst[k] = ESS_univariate(rh[::2, 0])     # ESS for theta
        rh_essu[k] = ESS_univariate(rh[::2, 1])     # ESS for u
        rh_essj[k] = ESS(rh[::2])                   # ESS joint
        rh_rmse[k] = sqrt(mean((target.logpdf(rh) - z0)**2))  # RMSE on energy
        rh_uniq[k] = n_unique(rh)                             # Number of unique samples
        rh_act[k] = acf(rh[::2, 0], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for theta
        rh_acu[k] = acf(rh[::2, 1], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for u
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
        'RH': {
            'A1': arh1,
            'A2': arh2,
            'E': erh,
            'EG': egrh, 
            'ET': etrh, 
            'ESS_T': rh_esst,
            'ESS_U': rh_essu,
            'ESS_J': rh_essj,
            'RMSE': rh_rmse,
            'UNIQUE': rh_uniq,
            'AC_T': rh_act,
            'AC_U': rh_acu
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

    Ts = [10, 1, 0.1, 0.01]
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

    # RotatedHUG
    THETA_ESS_RHUG = zeros((n_runs, n_epsilons, n_T, n_alphas))      
    U_ESS_RHUG = zeros((n_runs, n_epsilons, n_T, n_alphas))  
    ESS_JOINT_RHUG = zeros((n_runs, n_epsilons, n_T, n_alphas))        
    A_RHUG = zeros((n_runs, n_epsilons, n_T, n_alphas))            
    RMSE_RHUG = zeros((n_runs, n_epsilons, n_T, n_alphas))    
    EJSD_RHUG = zeros((n_runs, n_epsilons, n_T, n_alphas))             
    G_EJSD_RHUG = zeros((n_runs, n_epsilons, n_T, n_alphas))           
    T_EJSD_RHUG = zeros((n_runs, n_epsilons, n_T, n_alphas))
    A_HOP_RHUG = zeros((n_runs, n_epsilons, n_T, n_alphas))    
    N_UNIQUE_RHUG = zeros((n_runs, n_epsilons, n_T, n_alphas))     
    THETA_AC_RHUG = zeros((n_runs, n_epsilons, n_T, n_alphas, nlags)) 
    U_AC_RHUG = zeros((n_runs, n_epsilons, n_T, n_alphas, nlags)) 

    initial_time = time.time()
    for i in range(n_runs):
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
                # Store RotatedHUG results
                THETA_ESS_RHUG[i, j, k, :]   = out['RH']['ESS_T']
                U_ESS_RHUG[i, j, k, :]       = out['RH']['ESS_U']
                ESS_JOINT_RHUG[i, j, k, :]   = out['RH']['ESS_J']
                A_RHUG[i, j, k, :]           = out['RH']['A1']
                A_HOP_RHUG[i, j, k, :]       = out['RH']['A2']
                RMSE_RHUG[i, j, k, :]        = out['RH']['RMSE']
                EJSD_RHUG[i, j, k, :]        = out['RH']['E'] 
                G_EJSD_RHUG[i, j, k, :]      = out['RH']['EG'] 
                T_EJSD_RHUG[i, j, k, :]      = out['RH']['ET']
                N_UNIQUE_RHUG[i, j, k, :]    = out['RH']['UNIQUE']
                THETA_AC_RHUG[i, j, k, :, :] = out['RH']['AC_T']
                U_AC_RHUG[i, j, k, :, :]     = out['RH']['AC_U']

    # Save results
    folder = "experiment25/"

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

    save(folder + "THETA_ESS_RHUG.npy", THETA_ESS_RHUG)
    save(folder + "U_ESS_RHUG.npy", U_ESS_RHUG)
    save(folder + "ESS_JOINT_RHUG.npy", ESS_JOINT_RHUG)
    save(folder + "A_RHUG.npy", A_RHUG)
    save(folder + "RMSE_RHUG.npy", RMSE_RHUG)
    save(folder + "EJSD_RHUG.npy", EJSD_RHUG)
    save(folder + "G_EJSD_RHUG.npy", G_EJSD_RHUG)
    save(folder + "T_EJSD_RHUG.npy", T_EJSD_RHUG)
    save(folder + "A_HOP_RHUG.npy", A_HOP_RHUG)
    save(folder + "N_UNIQUE_RHUG.npy", N_UNIQUE_RHUG)
    save(folder + "THETA_AC_RHUG.npy", THETA_AC_RHUG)
    save(folder + "U_AC_RHUG.npy", U_AC_RHUG)

