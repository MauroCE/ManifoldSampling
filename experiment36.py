# Experiment 36: Similar to Experiment35 but we use a combination of the gradient of f
# and the gradient of \pi to reflect the velocity. Compare HUG and THUG using the 
# Epanechnikov kernel, mixture of gaussians prior.
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save, exp, linspace, pi
from numpy.linalg import solve, norm
from scipy.stats import multivariate_normal as MVN
from tangential_hug_functions import HugTangentialStepEJSD_Deterministic, HugStepEJSD_Deterministic
from utils import ESS_univariate, ESS, n_unique
from numpy.random import normal, rand, uniform
from statsmodels.tsa.stattools import acf
from Manifolds.RotatedEllipse import RotatedEllipse
import time
import math


def log_epanechnikov_kernel(xi, epsilon, target, z0):
    u = norm(target.logpdf(xi) - z0)
    with np.errstate(divide='ignore'):
        return log((3*(1 - (u**2 / (epsilon**2))) / (4*epsilon)) * float(norm(target.logpdf(xi) - z0) <= epsilon))
        

def log_epanechnikov_kernel_broadcast(xi_matrix, epsilon, target, z0):
    u_vector = sqrt((target.logpdf(xi_matrix) - z0)**2)
    with np.errstate(divide='ignore'):
        return log((3*(1 - (u_vector**2 / (epsilon**2))) / (4*epsilon)) * (u_vector <= epsilon).astype('float'))

def logprior_mixture(xi, mus, Sigmas, coefs):
    logpdf = 0.0
    for (mu, Sigma, c) in zip(mus, Sigmas, coefs):
        logpdf += c * MVN(mu, Sigma).logpdf(xi)
    return logpdf

def logprior_mixture_broadcast(xi_matrix, mus, Sigmas, coefs):
    n_samples = xi_matrix.shape[0]
    logpdf = zeros(n_samples)
    for i in range(n_samples):
        for (mu, Sigma, c) in zip(mus, Sigmas, coefs):
            logpdf[i] += c * MVN(mu, Sigma).logpdf(xi_matrix[i, :])
    return logpdf


def log_abc_posterior(xi, epsilon, target, z0, mus, Sigmas, coefs):
    """Log density of ABC posterior. Product of (param-latent) prior and uniform kernel."""
    return logprior_mixture(xi, mus, Sigmas, coefs) + log_epanechnikov_kernel(xi, epsilon, target, z0)


def grad_log_simulator(xi, Sigma):
    """Gradient of log simulator N(mu, Sigma)."""
    return - solve(Sigma, xi)

def gradient(xi, Sigma, mus, Sigmas, coefs):
    """"KEY DIFFERENCE FROM EXPERIMENT 35: Here the gradient
    is a combination of the gradient of f and the gradient of \pi."""
    gf = -solve(Sigma, xi)  # Gradient of the function f.
    ### COMPUTE GRADIENT OF PI.
    gaussians = [MVN(mu, Sigma) for (mu, Sigma) in zip(mus, Sigmas)] # Components of the mixture
    MG = lambda xy: np.sum(vstack([c * MVN.pdf(xy)  for (c, MVN) in zip(coefs, gaussians)]), axis=0) # Computes PDF value of mixture
    gp_func = lambda xy: (1 / MG(xy)) * np.sum(np.vstack([- c * MVN(mu, Sigma).pdf(xy) * solve(Sigma, xy - mu) for (c, mu, Sigma) in zip(coefs, mus, Sigmas)]), axis=0)
    return gf + gp_func(xi)


def check(x, name):
    """This way ESS will always work."""
    if type(x) == float or type(x) == np.float64:
        if math.isnan(x) or math.isinf(x):
            print("NaN found.")
            return 0.0
        else:
            return float(x)   # Works
    elif type(x) == np.ndarray:
        if np.isnan(x).any():
            print("NaN found.")
            x[np.isnan(x)] = 0.0
        if np.iscomplex(x).any():
            print("Complex found.")
            x[np.iscomplex(x)] = 0.0
        return x
    elif type(x) == complex:
        print(name, ": ", x)
        return 0.0
    else:
        print(name, ": ", x, " type: ", type(x))
        return 0.0


def experiment(x00, T, B, N, epsilon, alphas, q, target, z0, mus, Sigmas, coefs):
    """Runs Hug+Hop and THUG+HOP using the same velocities and the same random seeds.
    We also try to limit the noise in the HOP kernel by sampling the u variables beforehand.
    I run THUG for all values of alpha with the randomness fixed. 
    This is 1 run, for 1 epsilon. It does 1 HUG+HOP and then THUG+HOP for all alphas.
    T1: T for HUG
    T2: T for THUG
    """
    ### COMMON VARIABLES
    v = q.rvs(N)
    log_uniforms = log(rand(N))     # Log uniforms for the HUG kernels
    n_alphas = len(alphas)
    ### STORAGE (HUG + HOP)
    hh = x00              # Initial sample
    ahh = 0.0       # Acceptance probability for HUG kernel
    ehh = 0.0             # EJSD 
    eghh = 0.0            # EJSD in Gradient direction 
    ethh = 0.0            # EJSD in Tangent direction 
    ### STORAGE (THUG + HOP) I MUST STORE FOR ALL ALPHAS
    ath = zeros(n_alphas)
    eth  = zeros(n_alphas)
    egth = zeros(n_alphas)
    etth = zeros(n_alphas)
    ### ADDITIONAL STORAGE FOR THUG
    th_esst = zeros(n_alphas)
    th_essu = zeros(n_alphas)
    th_essj = zeros(n_alphas)
    th_ess_logpi = zeros(n_alphas)
    th_rmse = zeros(n_alphas)
    th_uniq = zeros(n_alphas)
    ### Redefine functions
    log_kernel           = lambda xi, epsilon: log_epanechnikov_kernel(xi, epsilon, target, z0)
    log_kernel_broadcast = lambda xi_matrix, epsilon: log_epanechnikov_kernel_broadcast(xi_matrix, epsilon, target, z0)
    log_post             = lambda xi: logprior_mixture(xi, mus, Sigmas, coefs) + log_kernel(xi, epsilon)
    log_post_broadcast   = lambda xi_matrix: logprior_mixture_broadcast(xi_matrix, mus, Sigmas, coefs) + log_kernel_broadcast(xi_matrix, epsilon)
    grad_log_sim = lambda xi: gradient(xi, target.cov, mus, Sigmas, coefs)
    ### HUG + HOP
    x = x00
    for i in range(N):
        x, a, e, eg, et = HugStepEJSD_Deterministic(x, v[i], log_uniforms[i], T, B, q, log_post, grad_log_sim)
        hh = vstack((hh, x))
        ahh += a * 100 / N
        ehh += e / N
        eghh += eg / N 
        ethh += et / N 
    # COMPUTE ESS AND OTHER METRICS FOR HUG
    hh = hh[1:]
    hh_esst = check(ESS_univariate(hh[:, 0]), "T ESS HUG")     # ESS for theta (Hug)
    hh_essu = check(ESS_univariate(hh[:, 1]), "U ESS HUG")    # ESS for u     (Hug)
    hh_ess_logpi = check(ESS_univariate(log_post_broadcast(hh)), "LOGPI ESS HUG") # ESS on logpi (Hug)
    hh_essj = check(ESS(hh), "J ESS HUG")                   # ESS joint     (Hug)
    hh_rmse = sqrt(mean((target.logpdf(hh) - z0)**2))  # RMSE on energy
    hh_uniq = n_unique(hh)                             # Number of unique samples
    ### THUG + HOP
    for k, alpha in enumerate(alphas):
        x = x00
        th = x00      # RESTART THE SAMPLES FROM SCRATCH
        for i in range(N):
            x, a, e, eg, et = HugTangentialStepEJSD_Deterministic(x, v[i], log_uniforms[i], T, B, alpha, q, log_post, grad_log_sim)
            th = vstack((th, x))
            ath[k] += a * 100 / N
            eth[k]  += e / N
            egth[k] += eg / N 
            etth[k] += et / N 
        ### COMPUTE ESS AND OTHER METRISC FOR THUG
        th = th[1:]
        th_esst[k] = check(ESS_univariate(th[:, 0]), "T ESS THUG")     # ESS for theta (Thug)
        th_essu[k] = check(ESS_univariate(th[:, 1]), "U ESS THUG")     # ESS for u     (Thug)
        th_ess_logpi[k] = check(ESS_univariate(log_post_broadcast(th)), "ESS LOGPI TH")
        th_essj[k] = check(ESS(th), "J ESS THUG")                   # ESS joint     (Thug)
        th_rmse[k] = sqrt(mean((target.logpdf(th) - z0)**2))  # RMSE on energy
        th_uniq[k] = n_unique(th)                             # Number of unique samples
    # RETURN EVERYTHING
    out = {
        'HH': {
            'A': ahh,
            'E': ehh,
            'EG': eghh, 
            'ET': ethh,
            'ESS_T': hh_esst,
            'ESS_U': hh_essu,
            'ESS_J': hh_essj,
            'ESS_LOGPI': hh_ess_logpi,
            'RMSE': hh_rmse,
            'UNIQUE': hh_uniq
        },
        'TH': {
            'A': ath,
            'E': eth,
            'EG': egth, 
            'ET': etth, 
            'ESS_T': th_esst,
            'ESS_U': th_essu,
            'ESS_J': th_essj,
            'ESS_LOGPI': th_ess_logpi,
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
    n_runs = 20 #15
    nlags = 20

    Ts = [10, 1, 0.1] #[7, 5, 3, 1, 0.1, 0.01]
    epsilons = [0.1, 0.001, 0.00001, 0.00000001]#0.0000001]
    alphas = [0.9, 0.99]
    n_epsilons = len(epsilons)
    n_alphas = len(alphas)
    n_T = len(Ts)

    # HUG
    THETA_ESS_HUG = zeros((n_runs, n_epsilons, n_T))        # Univariate ESS for \theta chain
    U_ESS_HUG = zeros((n_runs, n_epsilons, n_T))            # Univariate ESS for u chain
    LOGPI_ESS_HUG = zeros((n_runs, n_epsilons, n_T))        # Univariate ESS for logpi(xi) chain
    ESS_JOINT_HUG = zeros((n_runs, n_epsilons, n_T))        # multiESS on whole chain
    A_HUG = zeros((n_runs, n_epsilons, n_T))                # Acceptance probability
    RMSE_HUG = zeros((n_runs, n_epsilons, n_T))             # Root Mean Squared Error
    EJSD_HUG = zeros((n_runs, n_epsilons, n_T))             # Full EJSD 
    G_EJSD_HUG = zeros((n_runs, n_epsilons, n_T))           # EJSD for gradient only
    T_EJSD_HUG = zeros((n_runs, n_epsilons, n_T))           # EJSD for tangent only
    N_UNIQUE_HUG = zeros((n_runs, n_epsilons, n_T))         # Number of unique samples

    # THUG
    THETA_ESS_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))      
    U_ESS_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))  
    ESS_JOINT_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))   
    LOGPI_ESS_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))  # Univariate ESS for logpi(xi) chain.  
    A_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))            
    RMSE_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))    
    EJSD_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))             
    G_EJSD_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))           
    T_EJSD_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))
    N_UNIQUE_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))     

    initial_time = time.time()
    for i in range(n_runs):
        print("Run: ", i)
        # We need a new point for each run, but then must be the same for all other settings
        initial_point = new_point()
        for j, epsilon in enumerate(epsilons):
            lam = epsilon # For HOP
            for k, T in enumerate(Ts):
                out = experiment(initial_point, T, B, N, epsilon, alphas, q, target, z0, mus, Sigmas, coefs)

                # Store HUG results
                THETA_ESS_HUG[i, j, k]   = check(out['HH']['ESS_T'], "T ESS HUG")
                U_ESS_HUG[i, j, k]       = check(out['HH']['ESS_U'], "U ESS HUG")
                ESS_JOINT_HUG[i, j, k]   = check(out['HH']['ESS_J'], "J ESS HUG")
                LOGPI_ESS_HUG[i, j, k]   = check(out['HH']['ESS_LOGPI'], "LOGPI ESS HUG")
                A_HUG[i, j, k]           = out['HH']['A']
                RMSE_HUG[i, j, k]        = out['HH']['RMSE']
                EJSD_HUG[i, j, k]        = out['HH']['E']
                G_EJSD_HUG[i, j, k]      = out['HH']['EG']
                T_EJSD_HUG[i, j, k]      = out['HH']['ET']
                N_UNIQUE_HUG[i, j, k]    = out['HH']['UNIQUE']
                # Store THUG results
                THETA_ESS_THUG[i, j, k, :]   = check(out['TH']['ESS_T'], "T ESS THUG")
                U_ESS_THUG[i, j, k, :]       = check(out['TH']['ESS_U'], "U ESS THUG")
                ESS_JOINT_THUG[i, j, k, :]   = check(out['TH']['ESS_J'], "J ESS THUG")
                LOGPI_ESS_THUG[i, j, k, :]   = check(out['TH']['ESS_LOGPI'], "LOGPI ESS THUG")
                A_THUG[i, j, k, :]           = out['TH']['A']
                RMSE_THUG[i, j, k, :]        = out['TH']['RMSE']
                EJSD_THUG[i, j, k, :]        = out['TH']['E'] 
                G_EJSD_THUG[i, j, k, :]      = out['TH']['EG'] 
                T_EJSD_THUG[i, j, k, :]      = out['TH']['ET']
                N_UNIQUE_THUG[i, j, k, :]    = out['TH']['UNIQUE']

    # Save results
    folder = "experiment36/"  # same as 35 but now ess_logpi has been fixed

    save(folder + "EPSILONS.npy", epsilons)
    save(folder + "ALPHAS.npy", alphas)
    save(folder + "TS.npy", Ts)
    save(folder + "TIME.npy", np.array([time.time() - initial_time]))

    save(folder + "THETA_ESS_HUG.npy", THETA_ESS_HUG)
    save(folder + "U_ESS_HUG.npy", U_ESS_HUG)
    save(folder + "ESS_JOINT_HUG.npy", ESS_JOINT_HUG)
    save(folder + "LOGPI_ESS_HUG.npy", LOGPI_ESS_HUG)
    save(folder + "A_HUG.npy", A_HUG)
    save(folder + "RMSE_HUG.npy", RMSE_HUG)
    save(folder + "EJSD_HUG.npy", EJSD_HUG)
    save(folder + "G_EJSD_HUG.npy", G_EJSD_HUG)
    save(folder + "T_EJSD_HUG.npy", T_EJSD_HUG)
    save(folder + "N_UNIQUE_HUG.npy", N_UNIQUE_HUG)

    save(folder + "THETA_ESS_THUG.npy", THETA_ESS_THUG)
    save(folder + "U_ESS_THUG.npy", U_ESS_THUG)
    save(folder + "ESS_JOINT_THUG.npy", ESS_JOINT_THUG)
    save(folder + "LOGPI_ESS_THUG.npy", LOGPI_ESS_THUG)
    save(folder + "A_THUG.npy", A_THUG)
    save(folder + "RMSE_THUG.npy", RMSE_THUG)
    save(folder + "EJSD_THUG.npy", EJSD_THUG)
    save(folder + "G_EJSD_THUG.npy", G_EJSD_THUG)
    save(folder + "T_EJSD_THUG.npy", T_EJSD_THUG)
    save(folder + "N_UNIQUE_THUG.npy", N_UNIQUE_THUG)

