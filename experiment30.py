# Experiment 30: HUG+HOP vs THUG+HOP. Epanechnikov kernel, mixture of gaussians prior.
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save, exp, linspace, pi
from numpy.linalg import solve, norm
from scipy.stats import multivariate_normal as MVN
from tangential_hug_functions import HugTangentialStepEJSD_Deterministic, Hop_Deterministic, HugStepEJSD_Deterministic
from utils import ESS_univariate, ESS, n_unique
from numpy.random import normal, rand, uniform
from statsmodels.tsa.stattools import acf
from Manifolds.RotatedEllipse import RotatedEllipse
import time


def log_epanechnikov_kernel(xi, epsilon, target, z0):
    u = norm(target.logpdf(xi) - z0)
    with np.errstate(divide='ignore'):
        return log((3*(1 - (u**2 / (epsilon**2))) / (4*epsilon)) * float(norm(target.logpdf(xi) - z0) <= epsilon))
        

def logprior_mixture(xi, mus, Sigmas, coefs):
    logpdf = 0.0
    for (mu, Sigma, c) in zip(mus, Sigmas, coefs):
        logpdf += c * MVN(mu, Sigma).logpdf(xi)
    return logpdf


def log_abc_posterior(xi, epsilon, target, z0, mus, Sigmas, coefs):
    """Log density of ABC posterior. Product of (param-latent) prior and uniform kernel."""
    return logprior_mixture(xi, mus, Sigmas, coefs) + log_epanechnikov_kernel(xi, epsilon, target, z0)


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


def experiment(x00, T, B, N, lam, kappa, epsilon, alphas, q, target, z0, mus, Sigmas, coefs):
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
    n_alphas = len(alphas)
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
    th_essjtot = zeros(n_alphas)
    th_rmse = zeros(n_alphas)
    th_uniq = zeros(n_alphas)
    ### Redefine functions
    log_kernel = lambda xi, epsilon: log_epanechnikov_kernel(xi, epsilon, target, z0)
    log_post   = lambda xi: logprior_mixture(xi, mus, Sigmas, coefs) + log_kernel(xi, epsilon)
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
    hh_esst = ESS_univariate(hh[::2, 0])     # ESS for theta (Hug)
    hh_essu = ESS_univariate(hh[::2, 1])     # ESS for u     (Hug)
    hh_essj = ESS(hh[::2])                   # ESS joint     (Hug)
    hh_essjtot = ESS(hh)                     # ESS joint     (Hug + Hop)
    hh_rmse = sqrt(mean((target.logpdf(hh) - z0)**2))  # RMSE on energy
    hh_uniq = n_unique(hh)                             # Number of unique samples
    ### THUG + HOP
    for k, alpha in enumerate(alphas):
        x = x00
        th = x00      # RESTART THE SAMPLES FROM SCRATCH
        for i in range(N):
            y, a1, e, eg, et = HugTangentialStepEJSD_Deterministic(x, v[i], log_uniforms1[i], T, B, alpha, q, log_post, grad_log_sim)
            x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, log_post, grad_log_sim)
            th = vstack((th, y, x))
            ath1[k] += a1 * 100 / N
            ath2[k] += a2 * 100 / N
            eth[k]  += e / N
            egth[k] += eg / N 
            etth[k] += et / N 
        ### COMPUTE ESS AND OTHER METRISC FOR THUG
        th = th[1:]
        th_esst[k] = ESS_univariate(th[::2, 0])     # ESS for theta (Thug)
        th_essu[k] = ESS_univariate(th[::2, 1])     # ESS for u     (Thug)
        th_essj[k] = ESS(th[::2])                   # ESS joint     (Thug)
        th_essjtot[k] = ESS(th)                     # ESS joint     (Thug + Hop)
        th_rmse[k] = sqrt(mean((target.logpdf(th) - z0)**2))  # RMSE on energy
        th_uniq[k] = n_unique(th)                             # Number of unique samples
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
            'ESS_J_TOT': hh_essjtot,
            'RMSE': hh_rmse,
            'UNIQUE': hh_uniq
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
            'ESS_J_TOT': th_essjtot,
            'RMSE': th_rmse,
            'UNIQUE': th_uniq
        }
    }
    return out

        

if __name__ == "__main__":
    # Target distribution is a diagonal MVN
    Sigma = diag([1.0, 5.0])  # theta, u
    target = MVN(zeros(2), Sigma)

    # Proposal for velocity in HUG/THUG
    q = MVN(zeros(2), eye(2))

    # Means and Covariance Matrices for mixture prior
    mus = [
        np.array([0.5, 0.5]),
        np.array([0.0, -0.5]),
        np.array([-0.5, 0.0])
    ]
    Sigmas = [
        np.array([[1.0, 0.8], [0.8, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.5, 0.225], [0.225, 0.5]])
    ]
    coefs = np.repeat(1/len(mus), len(mus))

    # Sample a point at the beginning just to obtain some energy level
    z0 = -2.9513586307684885 #target.logpdf(target.rvs())
    # Function to sample a new point on the contour
    new_point = lambda: RotatedEllipse(zeros(2), Sigma, exp(z0)).to_cartesian(uniform(0, 2*pi))

    # Settings
    B = 5 
    N = 20000
    kappa = 0.25    
    n_runs = 10 #15
    nlags = 20

    Ts = [10, 1, 0.1] #[7, 5, 3, 1, 0.1, 0.01]
    epsilons = [0.1, 0.001, 0.00001, 0.0000001]
    alphas = [0.9, 0.99]
    n_epsilons = len(epsilons)
    n_alphas = len(alphas)
    n_T = len(Ts)

    # HUG
    THETA_ESS_HUG = zeros((n_runs, n_epsilons, n_T))        # Univariate ESS for \theta chain
    U_ESS_HUG = zeros((n_runs, n_epsilons, n_T))            # Univariate ESS for u chain
    ESS_JOINT_HUG = zeros((n_runs, n_epsilons, n_T))        # multiESS on whole chain
    ESS_JOINT_TOT_HUG = zeros((n_runs, n_epsilons, n_T))    # same as above but for hug and hop
    A_HUG = zeros((n_runs, n_epsilons, n_T))                # Acceptance probability
    RMSE_HUG = zeros((n_runs, n_epsilons, n_T))             # Root Mean Squared Error
    EJSD_HUG = zeros((n_runs, n_epsilons, n_T))             # Full EJSD 
    G_EJSD_HUG = zeros((n_runs, n_epsilons, n_T))           # EJSD for gradient only
    T_EJSD_HUG = zeros((n_runs, n_epsilons, n_T))           # EJSD for tangent only
    A_HOP_HUG  = zeros((n_runs, n_epsilons, n_T))           # Acceptance probability of HOP for HUG.
    N_UNIQUE_HUG = zeros((n_runs, n_epsilons, n_T))         # Number of unique samples

    # THUG
    THETA_ESS_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))      
    U_ESS_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))  
    ESS_JOINT_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))     
    ESS_JOINT_TOT_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))       
    A_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))            
    RMSE_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))    
    EJSD_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))             
    G_EJSD_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))           
    T_EJSD_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))
    A_HOP_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))    
    N_UNIQUE_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))     

    initial_time = time.time()
    for i in range(n_runs):
        print("Run: ", i)
        # We need a new point for each run, but then must be the same for all other settings
        initial_point = new_point()
        for j, epsilon in enumerate(epsilons):
            lam = epsilon # For HOP
            for k, T in enumerate(Ts):
                out = experiment(initial_point, T, B, N, lam, kappa, epsilon, alphas, q, target, z0, mus, Sigmas, coefs)

                # Store HUG results
                THETA_ESS_HUG[i, j, k]   = out['HH']['ESS_T']
                U_ESS_HUG[i, j, k]       = out['HH']['ESS_U']
                ESS_JOINT_HUG[i, j, k]   = out['HH']['ESS_J']
                ESS_JOINT_TOT_HUG[i, j, k] = out['HH']['ESS_J_TOT']
                A_HUG[i, j, k]           = out['HH']['A1']
                A_HOP_HUG[i, j, k]       = out['HH']['A2']
                RMSE_HUG[i, j, k]        = out['HH']['RMSE']
                EJSD_HUG[i, j, k]        = out['HH']['E']
                G_EJSD_HUG[i, j, k]      = out['HH']['EG']
                T_EJSD_HUG[i, j, k]      = out['HH']['ET']
                N_UNIQUE_HUG[i, j, k]    = out['HH']['UNIQUE']
                # Store THUG results
                THETA_ESS_THUG[i, j, k, :]   = out['TH']['ESS_T']
                U_ESS_THUG[i, j, k, :]       = out['TH']['ESS_U']
                ESS_JOINT_THUG[i, j, k, :]   = out['TH']['ESS_J']
                ESS_JOINT_TOT_THUG[i, j, k, :] = out['TH']['ESS_J_TOT']
                A_THUG[i, j, k, :]           = out['TH']['A1']
                A_HOP_THUG[i, j, k, :]       = out['TH']['A2']
                RMSE_THUG[i, j, k, :]        = out['TH']['RMSE']
                EJSD_THUG[i, j, k, :]        = out['TH']['E'] 
                G_EJSD_THUG[i, j, k, :]      = out['TH']['EG'] 
                T_EJSD_THUG[i, j, k, :]      = out['TH']['ET']
                N_UNIQUE_THUG[i, j, k, :]    = out['TH']['UNIQUE']

    # Save results
    folder = "dumper3/" #"experiment30/"

    save(folder + "EPSILONS.npy", epsilons)
    save(folder + "ALPHAS.npy", alphas)
    save(folder + "TS.npy", Ts)
    save(folder + "TIME.npy", np.array([time.time() - initial_time]))

    save(folder + "THETA_ESS_HUG.npy", THETA_ESS_HUG)
    save(folder + "U_ESS_HUG.npy", U_ESS_HUG)
    save(folder + "ESS_JOINT_HUG.npy", ESS_JOINT_HUG)
    save(folder + "ESS_JOINT_TOT_HUG.npy", ESS_JOINT_TOT_HUG)
    save(folder + "A_HUG.npy", A_HUG)
    save(folder + "RMSE_HUG.npy", RMSE_HUG)
    save(folder + "EJSD_HUG.npy", EJSD_HUG)
    save(folder + "G_EJSD_HUG.npy", G_EJSD_HUG)
    save(folder + "T_EJSD_HUG.npy", T_EJSD_HUG)
    save(folder + "A_HOP_HUG.npy", A_HOP_HUG)
    save(folder + "N_UNIQUE_HUG.npy", N_UNIQUE_HUG)

    save(folder + "THETA_ESS_THUG.npy", THETA_ESS_THUG)
    save(folder + "U_ESS_THUG.npy", U_ESS_THUG)
    save(folder + "ESS_JOINT_THUG.npy", ESS_JOINT_THUG)
    save(folder + "ESS_JOINT_TOT_THUG.npy", ESS_JOINT_TOT_THUG)
    save(folder + "A_THUG.npy", A_THUG)
    save(folder + "RMSE_THUG.npy", RMSE_THUG)
    save(folder + "EJSD_THUG.npy", EJSD_THUG)
    save(folder + "G_EJSD_THUG.npy", G_EJSD_THUG)
    save(folder + "T_EJSD_THUG.npy", T_EJSD_THUG)
    save(folder + "A_HOP_THUG.npy", A_HOP_THUG)
    save(folder + "N_UNIQUE_THUG.npy", N_UNIQUE_THUG)

