# Experiment 29c: Hug vs Hug+Hop on Gaussian kernel Gaussian prior.
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save, exp, linspace, pi
from numpy.linalg import solve
from scipy.stats import multivariate_normal as MVN
from tangential_hug_functions import Hop_Deterministic, HugStepEJSD_Deterministic
from utils import ESS_univariate, ESS, n_unique
from numpy.random import normal, rand, uniform
from statsmodels.tsa.stattools import acf
from Manifolds.RotatedEllipse import RotatedEllipse
import time


def log_normal_kernel(xi, epsilon, target, z0):
    """Log density of normal kernel."""
    return MVN(z0, epsilon**2).logpdf(target.logpdf(xi))


def logprior_normal(xi):
    """Log density for normal prior p(xi) of parameters and latents N(0, I)."""
    return MVN(zeros(2), eye(2)).logpdf(xi)


def log_abc_posterior(xi, epsilon, target, z0):
    """Log density of ABC posterior. Product of (param-latent) prior and uniform kernel."""
    return logprior_normal(xi) + log_normal_kernel(xi, epsilon, target, z0)


def grad_log_simulator(xi, Sigma):
    """Gradient of log simulator N(mu, Sigma)."""
    return - solve(Sigma, xi)


def check(x, name):
    """This way ESS will always work."""
    if type(x) == float or type(x) == np.float64:
        return float(x)
    elif type(x) == complex:
        print(name, ": ", x)
        return 0.0
    else:
        print(name, ": ", x, " type: ", type(x), " floated: ", float(x))
        return 0.0


def experiment(x00, T, B, N, lam, kappa, epsilon, q, target, z0, nlags):
    """Runs Hug+Hop and Hug using the same velocities and the same random seeds."""
    ### COMMON VARIABLES
    v = q.rvs(N)
    log_uniforms1 = log(rand(N))     # Log uniforms for the HUG kernels
    log_uniforms2 = log(rand(N))     # Log uniforms for the HOP kernel
    u = MVN(zeros(2), eye(2)).rvs(N) # Original velocities for HOP kernel
    ### STORAGE (HUG + HOP)
    hh = x00         # Initial sample
    ahh1 = 0.0       # Acceptance probability for HUG kernel
    ahh2 = 0.0       # Acceptance probability for HOP kernel (when used with HUG)
    ehh = 0.0             # EJSD
    eghh = 0.0            # EJSD in Gradient direction
    ethh = 0.0            # EJSD in Tangent direction
    ### STORAGE (HUG only)
    ho = x00 
    aho1 = 0.0
    eho  = 0.0
    egho = 0.0
    etho = 0.0
    ### Redefine functions
    log_kernel = lambda xi, epsilon: log_normal_kernel(xi, epsilon, target, z0)
    log_post   = lambda xi: logprior_normal(xi) + log_kernel(xi, epsilon)
    grad_log_sim = lambda xi: grad_log_simulator(xi, target.cov)
    ### HUG + HOP
    x = x00
    for i in range(N):
        y, a1, e, eg, et = HugStepEJSD_Deterministic(x, v[i], log_uniforms1[i], T, B, q, log_post, grad_log_sim)
        x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, log_post, grad_log_sim)
        hh = vstack((hh, y, x))
        ahh1 += a1 * 100 / N
        ahh2 += a2 * 100 / N
        ehh += e / N
        eghh += eg / N 
        ethh += et / N 
    # COMPUTE ESS AND OTHER METRICS FOR HUG
    hh = hh[1:]
    hh_esst = ESS_univariate(hh[::2, 0])     # ESS for theta (from Hug samples only)
    hh_essu = ESS_univariate(hh[::2, 1])     # ESS for u     (from Hug samples only)
    hh_essj = ESS(hh[::2])                   # ESS joint     (frmo Hug samples only)
    hh_rmse = sqrt(mean((target.logpdf(hh) - z0)**2))  # RMSE on energy
    hh_uniq = n_unique(hh)                             # Number of unique samples
    # hh_act  = acf(hh[::2, 0], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for theta (remove the first 1.0)
    # hh_acu  = acf(hh[::2, 1], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for u

    ### HUG only
    x = x00
    v = np.vstack((v, q.rvs(N)))
    log_uniforms1 = np.hstack((log_uniforms1, log_uniforms2))
    for i in range(2*N):
        x, a1, e, eg, et = HugStepEJSD_Deterministic(x, v[i], log_uniforms1[i], T, B, q, log_post, grad_log_sim)
        ho = vstack((ho, x))
        aho1 += a1 * 100 / (2*N)
        eho += e / (2*N)
        egho += eg / (2*N) 
        etho += et / (2*N) 
    # COMPUTE ESS AND OTHER METRICS FOR HUG
    ho = ho[1:]
    ho_esst = ESS_univariate(ho[::2, 0])     # ESS for theta
    ho_essu = ESS_univariate(ho[::2, 1])     # ESS for u
    ho_essj = ESS(ho[::2])                   # ESS joint
    ho_rmse = sqrt(mean((target.logpdf(ho) - z0)**2))  # RMSE on energy
    ho_uniq = n_unique(ho)                             # Number of unique samples
    # ho_act  = acf(ho[::2, 0], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for theta (remove the first 1.0)
    # ho_acu  = acf(ho[::2, 1], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for u

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
            'UNIQUE': hh_uniq
        },
        'HO': {
            'A1': aho1,
            'E': eho,
            'EG': egho, 
            'ET': etho, 
            'ESS_T': ho_esst,
            'ESS_U': ho_essu,
            'ESS_J': ho_essj,
            'RMSE': ho_rmse,
            'UNIQUE': ho_uniq
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
    N = 15000 #50000
    kappa = 0.25    
    n_runs = 10 #20 #15
    nlags = 20

    Ts = [10, 1, 0.1] #[10, 1, 0.1, 0.01] #[7, 5, 3, 1, 0.1, 0.01]
    epsilons = [0.1, 0.001, 0.00001, 0.0000001] #[0.1, 0.001, 0.00001, 0.0000001]
    n_epsilons = len(epsilons)
    n_T = len(Ts)

    # HUG
    THETA_ESS_HH = zeros((n_runs, n_epsilons, n_T))        # Univariate ESS for \theta chain
    U_ESS_HH = zeros((n_runs, n_epsilons, n_T))            # Univariate ESS for u chain
    ESS_JOINT_HH = zeros((n_runs, n_epsilons, n_T))        # multiESS on whole chain
    A_HH = zeros((n_runs, n_epsilons, n_T))                # Acceptance probability
    RMSE_HH = zeros((n_runs, n_epsilons, n_T))             # Root Mean Squared Error
    EJSD_HH = zeros((n_runs, n_epsilons, n_T))             # Full EJSD 
    G_EJSD_HH = zeros((n_runs, n_epsilons, n_T))           # EJSD for gradient only
    T_EJSD_HH = zeros((n_runs, n_epsilons, n_T))           # EJSD for tangent only
    A_HOP_HH  = zeros((n_runs, n_epsilons, n_T))           # Acceptance probability of HOP for HUG.
    N_UNIQUE_HH = zeros((n_runs, n_epsilons, n_T))         # Number of unique samples
    THETA_AC_HH = zeros((n_runs, n_epsilons, n_T, nlags))  # Autocorrelation for theta
    U_AC_HH = zeros((n_runs, n_epsilons, n_T, nlags))      # Autocorrelation for u

    # THUG
    THETA_ESS_HO = zeros((n_runs, n_epsilons, n_T))      
    U_ESS_HO = zeros((n_runs, n_epsilons, n_T))  
    ESS_JOINT_HO = zeros((n_runs, n_epsilons, n_T))        
    A_HO = zeros((n_runs, n_epsilons, n_T))            
    RMSE_HO = zeros((n_runs, n_epsilons, n_T))    
    EJSD_HO = zeros((n_runs, n_epsilons, n_T))             
    G_EJSD_HO = zeros((n_runs, n_epsilons, n_T))           
    T_EJSD_HO = zeros((n_runs, n_epsilons, n_T))
    N_UNIQUE_HO = zeros((n_runs, n_epsilons, n_T))     
    THETA_AC_HO = zeros((n_runs, n_epsilons, n_T, nlags)) 
    U_AC_HO = zeros((n_runs, n_epsilons, n_T, nlags)) 

    initial_time = time.time()
    for i in range(n_runs):
        # We need a new point for each run, but then must be the same for all other settings
        initial_point = new_point()
        for j, epsilon in enumerate(epsilons):
            lam = epsilon # For HOP
            for k, T in enumerate(Ts):
                out = experiment(initial_point, T, B, N, lam, kappa, epsilon, q, target, z0, nlags)
                # Store HUG & HOP results
                THETA_ESS_HH[i, j, k]   = check(out['HH']['ESS_T'], "ESS theta HH")
                U_ESS_HH[i, j, k]       = check(out['HH']['ESS_U'], "ESS u HH")
                ESS_JOINT_HH[i, j, k]   = check(out['HH']['ESS_J'], "multiESS HH")
                A_HH[i, j, k]           = out['HH']['A1']
                A_HOP_HH[i, j, k]       = out['HH']['A2']
                RMSE_HH[i, j, k]        = out['HH']['RMSE']
                EJSD_HH[i, j, k]        = out['HH']['E']
                G_EJSD_HH[i, j, k]      = out['HH']['EG']
                T_EJSD_HH[i, j, k]      = out['HH']['ET']
                N_UNIQUE_HH[i, j, k]    = out['HH']['UNIQUE']
                # Store HUG results
                THETA_ESS_HO[i, j, k]   = check(out['HO']['ESS_T'], "ESS theta HO")
                U_ESS_HO[i, j, k]       = check(out['HO']['ESS_U'], "ESS u HO")
                ESS_JOINT_HO[i, j, k]   = check(out['HO']['ESS_J'], "multiESS HH")
                A_HO[i, j, k]           = out['HO']['A1']
                RMSE_HO[i, j, k]        = out['HO']['RMSE']
                EJSD_HO[i, j, k]        = out['HO']['E']
                G_EJSD_HO[i, j, k]      = out['HO']['EG']
                T_EJSD_HO[i, j, k]      = out['HO']['ET']
                N_UNIQUE_HO[i, j, k]    = out['HO']['UNIQUE']

    print("Total time: ", time.time() - initial_time)

    # Save results
    folder = "experiment29c/" 

    save(folder + "EPSILONS.npy", epsilons)
    save(folder + "TS.npy", Ts)
    save(folder + "TIME.npy", np.array([time.time() - initial_time]))
    save(folder + "N.npy", N)
    save(folder + "N_RUNS.npy", n_runs)

    save(folder + "THETA_ESS_HH.npy", THETA_ESS_HH)
    save(folder + "U_ESS_HH.npy", U_ESS_HH)
    save(folder + "ESS_JOINT_HH.npy", ESS_JOINT_HH)
    save(folder + "A_HH.npy", A_HH)
    save(folder + "RMSE_HH.npy", RMSE_HH)
    save(folder + "EJSD_HH.npy", EJSD_HH)
    save(folder + "G_EJSD_HH.npy", G_EJSD_HH)
    save(folder + "T_EJSD_HH.npy", T_EJSD_HH)
    save(folder + "A_HOP_HH.npy", A_HOP_HH)
    save(folder + "N_UNIQUE_HH.npy", N_UNIQUE_HH)
    save(folder + "THETA_AC_HH.npy", THETA_AC_HH)
    save(folder + "U_AC_HH.npy", U_AC_HH)

    save(folder + "THETA_ESS_HO.npy", THETA_ESS_HO)
    save(folder + "U_ESS_HO.npy", U_ESS_HO)
    save(folder + "ESS_JOINT_HO.npy", ESS_JOINT_HO)
    save(folder + "A_HO.npy", A_HO)
    save(folder + "RMSE_HO.npy", RMSE_HO)
    save(folder + "EJSD_HO.npy", EJSD_HO)
    save(folder + "G_EJSD_HO.npy", G_EJSD_HO)
    save(folder + "T_EJSD_HO.npy", T_EJSD_HO)
    save(folder + "N_UNIQUE_HO.npy", N_UNIQUE_HO)
    save(folder + "THETA_AC_HO.npy", THETA_AC_HO)
    save(folder + "U_AC_HO.npy", U_AC_HO)

