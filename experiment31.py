# Experiment 31: HUG+HOP vs THUG+HOP. Gaussian example from "A rare-event approach to high-dimensional ABC".
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save, exp, linspace, pi
from numpy.linalg import solve, norm
from scipy.stats import multivariate_normal as MVN
from scipy.stats.stats import sigmaclip
from scipy.optimize import fsolve
from tangential_hug_functions import HugTangentialStepEJSD_Deterministic_Delta, Hop_Deterministic, HugStepEJSD_Deterministic
from utils import ESS_univariate, ESS, n_unique
from numpy.random import normal, rand, uniform, randn
from statsmodels.tsa.stattools import acf
from Manifolds.RotatedEllipse import RotatedEllipse
import time
from warnings import catch_warnings, filterwarnings


def f(xi): # Input R^n. Output R^{n-1}.
    """Simulator. Takes \\xi = (\\theta, z) and returns y. This is a deterministic function."""
    return xi[0] * xi[1:]

def f_broadcast(xi_matrix):
    """Computes f(xi) for each row of xi_matrix. Xi_matrix has dimension (N, n)."""
    return f(xi_matrix.T).T

def fnorm(xi, ystar): # Input R^n. Output R.
    """This function is h(\\xi) = \\|f(\\xi) - y*\\|. Basically the function defining Chang's manifold."""
    return norm(f(xi) - ystar)

def fnorm_broadcast(xi_matrix, y_star):
    """Same as fnorm but broadcasted to a whole matrix"""
    return norm(f_broadcast(xi_matrix) - y_star, axis=1)

def Jf_transpose(xi): # Input R^n. Output R^{n x m}.
    """Since f:Rn -> Rm the Jacobian is (m, n). This is the transpose of the Jacobian of f. """
    return np.vstack((xi[1:], xi[0] * eye(len(xi) - 1)))

def grad_fnorm(xi, ystar): # Input R^n. Output R^n.
    """This is the GRADIENT of h(\\xi) i.e. fnorm. Notice that epsilon goes away so it is not important."""
    return Jf_transpose(xi) @ (f(xi) - ystar)

def logprior(xi):
    return -log(10) + MVN(zeros(len(xi)-1), eye(len(xi)-1)).logpdf(xi[1:])

def logprior_broadcast(xi_matrix):
    """Same as logprior but broadcasted to a whole matrix."""
    nz = xi_matrix.shape[1] - 1
    return -log(10) + MVN(zeros(nz), eye(nz)).logpdf(xi_matrix[:, 1:])

def log_epanechnikov_kernel(xi, epsilon, fnorm, ystar):
    u = fnorm(xi, ystar)
    with np.errstate(divide='ignore'):
        return log((3*(1 - (u**2 / (epsilon**2))) / (4*epsilon)) * float(u <= epsilon))

def log_epanechnikov_kernel_broadcast(xi_matrix, epsilon, fnorm_broadcast, y_star):
    """Same as log_epanechnikov kernel but broadcasted to a matrix."""
    u = fnorm_broadcast(xi_matrix, y_star)
    with np.errstate(divide='ignore'):
        return log((3*(1 - (u**2 / (epsilon**2))) / (4*epsilon)) * (u <= epsilon).astype('float')) 

def log_abc_posterior(xi, epsilon, fnorm, ystar):
    """Log density of ABC posterior. Product of (param-latent) prior and Epanechnikov kernel."""
    return logprior(xi) + log_epanechnikov_kernel(xi, epsilon, fnorm, ystar)

def log_abc_posterior_broadcast(xi_matrix, epsilon):
    """Same as log_abc_posterior but broadcasted to a matrix."""
    return logprior_broadcast(xi_matrix) + log_epanechnikov_kernel_broadcast(xi_matrix, epsilon)

def check(x, name):
    """This way ESS will always work."""
    if type(x) == float or type(x) == np.float64:
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


def experiment(x00, T, B, N, lam, kappa, epsilon, alphas, q, fnorm, y_star):
    """Runs Hug+Hop and THUG+HOP using the same velocities and the same random seeds.
    We also try to limit the noise in the HOP kernel by sampling the u variables beforehand.
    I run THUG for all values of alpha with the randomness fixed. 
    This is 1 run, for 1 epsilon. It does 1 HUG+HOP and then THUG+HOP for all alphas.
    T1: T for HUG
    T2: T for THUG
    """
    ### COMMON VARIABLES
    d = len(x00)
    v = q.rvs(N)
    log_uniforms1 = log(rand(N))     # Log uniforms for the HUG kernels
    log_uniforms2 = log(rand(N))     # Log uniforms for the HOP kernel
    u = MVN(zeros(d), eye(d)).rvs(N) # Original velocities for HOP kernel
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
    dkth = zeros((n_alphas, N))  # delta k
    dlth = zeros((n_alphas, N))  # delta l
    afth = zeros((n_alphas, N))  # acceptance flag for THUG
    ### ADDITIONAL STORAGE FOR THUG
    th_esst = zeros(n_alphas)
    th_essu = zeros(n_alphas)
    th_essj = zeros(n_alphas)
    th_essjtot = zeros(n_alphas)
    th_ess_logpi = zeros(n_alphas)
    th_rmse = zeros(n_alphas)
    th_uniq = zeros(n_alphas)
    th_time = zeros(n_alphas)
    ### Redefine functions
    log_kernel = lambda xi, epsilon: log_epanechnikov_kernel(xi, epsilon, fnorm, y_star)
    log_kernel_broadcast = lambda xi, epsilon: log_epanechnikov_kernel_broadcast(xi, epsilon, fnorm_broadcast, y_star)
    log_post   = lambda xi: logprior(xi) + log_kernel(xi, epsilon)
    log_post_broadcast = lambda xi: logprior_broadcast(xi) + log_kernel_broadcast(xi, epsilon)
    grad_log_sim = lambda xi: grad_fnorm(xi, y_star)
    ### HUG + HOP
    hh_start = time.time()
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
    hh_essu = ESS(hh[::2, 1:])               # ESS for u     (Hug)
    hh_essj = ESS(hh[::2])                   # ESS joint     (Hug)
    hh_ess_logpi = ESS_univariate(log_post_broadcast(hh[::2])) # ESS on logpi (Hug)
    hh_essjtot = ESS(hh)                     # ESS joint     (Hug + Hop)
    hh_rmse = sqrt(mean((f_broadcast(hh) - y_star)**2)) # RMSE on energy
    hh_uniq = n_unique(hh)                             # Number of unique samples
    hh_time = time.time() - hh_start
    ### THUG + HOP
    for k, alpha in enumerate(alphas):
        th_start = time.time()
        x = x00
        th = x00      # RESTART THE SAMPLES FROM SCRATCH
        for i in range(N):
            y, a1, e, eg, et, dk, dl = HugTangentialStepEJSD_Deterministic_Delta(x, v[i], log_uniforms1[i], T, B, alpha, q, log_post, grad_log_sim)
            x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, log_post, grad_log_sim)
            th = vstack((th, y, x))
            dkth[k, i] = dk 
            dlth[k, i] = dl
            afth[k, i] = a1
            ath1[k] += a1 * 100 / N
            ath2[k] += a2 * 100 / N
            eth[k]  += e / N
            egth[k] += eg / N 
            etth[k] += et / N 
        th_time[k] = time.time() - th_start
        ### COMPUTE ESS AND OTHER METRISC FOR THUG
        th = th[1:]
        th_esst[k] = ESS_univariate(th[::2, 0])                      # ESS for theta (Thug)
        th_essu[k] = check(ESS(th[::2, 1:]), "Z ESS TH")             # ESS for u     (Thug)
        th_essj[k] = check(ESS(th[::2]), "ESS TH")                   # ESS joint     (Thug)
        th_ess_logpi[k] = check(ESS_univariate(log_post_broadcast(th[::2])), "ESS LOGPI TH")
        th_essjtot[k] = check(ESS(th), "ESS TOT TH")                 # ESS joint     (Thug + Hop)
        th_rmse[k] = sqrt(mean((f_broadcast(th) - y_star)**2))  # RMSE on energy
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
            'ESS_LOGPI': hh_ess_logpi,
            'ESS_J_TOT': hh_essjtot,
            'RMSE': hh_rmse,
            'UNIQUE': hh_uniq,
            'SAMPLES': hh,
            'TIME': hh_time
        },
        'TH': {
            'A1': ath1,
            'A2': ath2,
            'AF': afth,
            'E': eth,
            'EG': egth, 
            'ET': etth, 
            'ESS_T': th_esst,
            'ESS_U': th_essu,
            'ESS_J': th_essj,
            'ESS_LOGPI': th_ess_logpi,
            'ESS_J_TOT': th_essjtot,
            'RMSE': th_rmse,
            'UNIQUE': th_uniq,
            'SAMPLES': th,
            'DELTA_K': dkth,
            'DELTA_L': dlth,
            'TIME': th_time
        }
    }
    return out

        

if __name__ == "__main__":
    # Generate observed data
    nz = 25                               # Number of latents
    sigma_star, z_star = 3.0, randn(nz)   # Generate original \theta and z.
    xi_star = np.r_[sigma_star, z_star]
    y_star = f(xi_star)

    # Total dimensionality of \xi space is nz+1
    d = nz + 1

    # Proposal for velocity in HUG/THUG
    q = MVN(zeros(d), eye(d)) 

    # Need to find a point near the manifold to start with
    def new_point(nz, fnorm, y_star, d):
        with catch_warnings():
            filterwarnings('error')
            found = False
            while not found:
                try:
                    guess = np.r_[uniform(low=0.0, high=10.0), randn(nz)]  # Construct guess
                    func = lambda xi: np.r_[fnorm(xi, y_star), zeros(d-1)] # Construct function of which we want to find root
                    point = fsolve(func, guess)
                    found = True
                    return point
                except Warning:
                    continue

    # Settings
    B = 5 
    N = 10000
    kappa = 0.25    
    n_runs = 3 #15
    nlags = 20

    Ts = [5.0, 2.5, 1.0] #[0.5, 0.3, 0.1] #[10, 1, 0.1] #[7, 5, 3, 1, 0.1, 0.01]
    epsilons = [10.0, 5.0, 3.0, 2.0] #[0.1, 0.0001] #[0.1, 0.001, 0.00001, 0.0000001]
    alphas = [0.8, 0.9, 0.95] #[0.6, 0.7, 0.8, 0.9, 0.95]
    n_epsilons = len(epsilons)
    n_alphas = len(alphas)
    n_T = len(Ts)

    # HUG
    THETA_ESS_HUG = zeros((n_runs, n_epsilons, n_T))        # Univariate ESS for \theta chain
    U_ESS_HUG = zeros((n_runs, n_epsilons, n_T))            # Univariate ESS for u chain
    LOGPI_ESS_HUG = zeros((n_runs, n_epsilons, n_T))        # Univariate ESS for logpi(xi) chain
    ESS_JOINT_HUG = zeros((n_runs, n_epsilons, n_T))        # multiESS on whole chain
    ESS_JOINT_TOT_HUG = zeros((n_runs, n_epsilons, n_T))    # same as above but for hug and hop
    A_HUG = zeros((n_runs, n_epsilons, n_T))                # Acceptance probability
    RMSE_HUG = zeros((n_runs, n_epsilons, n_T))             # Root Mean Squared Error
    EJSD_HUG = zeros((n_runs, n_epsilons, n_T))             # Full EJSD 
    G_EJSD_HUG = zeros((n_runs, n_epsilons, n_T))           # EJSD for gradient only
    T_EJSD_HUG = zeros((n_runs, n_epsilons, n_T))           # EJSD for tangent only
    A_HOP_HUG  = zeros((n_runs, n_epsilons, n_T))           # Acceptance probability of HOP for HUG.
    N_UNIQUE_HUG = zeros((n_runs, n_epsilons, n_T))         # Number of unique samples
    TIME_HUG = zeros((n_runs, n_epsilons, n_T))             # Time taken by Hug for each iteration

    # THUG
    THETA_ESS_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))      
    U_ESS_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))  
    LOGPI_ESS_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))  # Univariate ESS for logpi(xi) chain.
    ESS_JOINT_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))     
    ESS_JOINT_TOT_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))       
    A_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))            
    RMSE_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))    
    EJSD_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))             
    G_EJSD_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))           
    T_EJSD_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))
    A_HOP_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))    
    N_UNIQUE_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))    
    DELTA_K = zeros((n_runs, n_epsilons, n_T, n_alphas, N)) 
    DELTA_L = zeros((n_runs, n_epsilons, n_T, n_alphas, N)) 
    A_FLAG_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas, N)) 
    TIME_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))

    initial_time = time.time()
    for i in range(n_runs):
        print("Run: ", i)
        # We need a new point for each run, but then must be the same for all other settings
        initial_point = new_point(nz, fnorm, y_star, d)
        for j, epsilon in enumerate(epsilons):
            lam = epsilon # For HOP
            for k, T in enumerate(Ts):
                out = experiment(initial_point, T, B, N, lam, kappa, epsilon, alphas, q, fnorm, y_star)

                # Store HUG results
                THETA_ESS_HUG[i, j, k]   = check(out['HH']['ESS_T'], "THETA ESS HH")
                U_ESS_HUG[i, j, k]       = check(out['HH']['ESS_U'], "U ESS HH")
                LOGPI_ESS_HUG[i, j, k]   = check(out['HH']['ESS_LOGPI'], "LOGPI ESS HH")
                ESS_JOINT_HUG[i, j, k]   = check(out['HH']['ESS_J'], "ESS HH")
                ESS_JOINT_TOT_HUG[i, j, k] = check(out['HH']['ESS_J_TOT'], "ESS HH TOT")
                A_HUG[i, j, k]           = out['HH']['A1']
                A_HOP_HUG[i, j, k]       = out['HH']['A2']
                RMSE_HUG[i, j, k]        = out['HH']['RMSE']
                EJSD_HUG[i, j, k]        = out['HH']['E']
                G_EJSD_HUG[i, j, k]      = out['HH']['EG']
                T_EJSD_HUG[i, j, k]      = out['HH']['ET']
                N_UNIQUE_HUG[i, j, k]    = out['HH']['UNIQUE']
                TIME_HUG[i, j, k]        = out['HH']["TIME"]

                # Store THUG results
                THETA_ESS_THUG[i, j, k, :]   = check(out['TH']['ESS_T'], "THETA ESS TH")
                U_ESS_THUG[i, j, k, :]       = check(out['TH']['ESS_U'], "U ESS TH")
                LOGPI_ESS_THUG[i, j, k, :]   = check(out['TH']['ESS_LOGPI'], "LOGPI ESS TH")
                ESS_JOINT_THUG[i, j, k, :]   = check(out['TH']['ESS_J'], "ESS TH")
                ESS_JOINT_TOT_THUG[i, j, k, :] = out['TH']['ESS_J_TOT']
                A_THUG[i, j, k, :]           = out['TH']['A1']
                A_HOP_THUG[i, j, k, :]       = out['TH']['A2']
                RMSE_THUG[i, j, k, :]        = out['TH']['RMSE']
                EJSD_THUG[i, j, k, :]        = out['TH']['E'] 
                G_EJSD_THUG[i, j, k, :]      = out['TH']['EG'] 
                T_EJSD_THUG[i, j, k, :]      = out['TH']['ET']
                N_UNIQUE_THUG[i, j, k, :]    = out['TH']['UNIQUE']
                DELTA_K[i, j, k, :, :]       = out['TH']['DELTA_K']
                DELTA_L[i, j, k, :, :]       = out['TH']['DELTA_L']
                A_FLAG_THUG[i, j, k, :, :]   = out['TH']['AF']
                TIME_THUG[i, j, k, :]        = out['TH']['TIME']
                

    # Save results
    folder = "experiment31/" #"experiment30/"

    save(folder + "EPSILONS.npy", epsilons)
    save(folder + "ALPHAS.npy", alphas)
    save(folder + "TS.npy", Ts)
    save(folder + "TIME.npy", np.array([time.time() - initial_time]))
    save(folder + "N.npy", np.array([N]))

    save(folder + "THETA_ESS_HUG.npy", THETA_ESS_HUG)
    save(folder + "U_ESS_HUG.npy", U_ESS_HUG)
    save(folder + "LOGPI_ESS_HUG.npy", LOGPI_ESS_HUG)
    save(folder + "ESS_JOINT_HUG.npy", ESS_JOINT_HUG)
    save(folder + "ESS_JOINT_TOT_HUG.npy", ESS_JOINT_TOT_HUG)
    save(folder + "A_HUG.npy", A_HUG)
    save(folder + "RMSE_HUG.npy", RMSE_HUG)
    save(folder + "EJSD_HUG.npy", EJSD_HUG)
    save(folder + "G_EJSD_HUG.npy", G_EJSD_HUG)
    save(folder + "T_EJSD_HUG.npy", T_EJSD_HUG)
    save(folder + "A_HOP_HUG.npy", A_HOP_HUG)
    save(folder + "N_UNIQUE_HUG.npy", N_UNIQUE_HUG)
    save(folder + "TIME_HUG.npy", TIME_HUG)

    save(folder + "THETA_ESS_THUG.npy", THETA_ESS_THUG)
    save(folder + "U_ESS_THUG.npy", U_ESS_THUG)
    save(folder + "LOGPI_ESS_THUG.npy", LOGPI_ESS_THUG)
    save(folder + "ESS_JOINT_THUG.npy", ESS_JOINT_THUG)
    save(folder + "ESS_JOINT_TOT_THUG.npy", ESS_JOINT_TOT_THUG)
    save(folder + "A_THUG.npy", A_THUG)
    save(folder + "RMSE_THUG.npy", RMSE_THUG)
    save(folder + "EJSD_THUG.npy", EJSD_THUG)
    save(folder + "G_EJSD_THUG.npy", G_EJSD_THUG)
    save(folder + "T_EJSD_THUG.npy", T_EJSD_THUG)
    save(folder + "A_HOP_THUG.npy", A_HOP_THUG)
    save(folder + "N_UNIQUE_THUG.npy", N_UNIQUE_THUG)
    save(folder + "DELTA_K.npy", DELTA_K)
    save(folder + "DELTA_L.npy", DELTA_L)
    save(folder + "A_FLAG_THUG.npy", A_FLAG_THUG)
    save(folder + "TIME_THUG.npy", TIME_THUG)

