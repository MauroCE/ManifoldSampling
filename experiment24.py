# Experiment 24: Uniform Kernel. Uniform Prior. HUG+HOP vs THUG+HOP when dimensionality increases.
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save, exp, linspace, pi
from numpy.linalg import solve
from scipy.stats import multivariate_normal as MVN
from tangential_hug_functions import HugTangentialStepEJSD_Deterministic, Hop_Deterministic, HugStepEJSD_Deterministic
from utils import ESS_univariate, ESS, n_unique
from numpy.random import normal, rand, uniform
from statsmodels.tsa.stattools import acf
from Manifolds.GeneralizedEllipse import GeneralizedEllipse
import time
from multiprocessing import Pool
import warnings
warnings.filterwarnings("error")
from itertools import product


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
    
def log_abc_posterior(xi, epsilon):
    """Log density of ABC posterior. Product of (param-latent) prior and uniform kernel."""
    return logprior_uniform(xi) + log_uniform_kernel(xi, epsilon)

def log_abc_posterior_all(xi):
    """Log density of ABC posterior. Product of (param-latent) prior and uniform kernel."""
    return logprior_uniform_all(xi) + log_uniform_kernel(xi, epsilon)
    
def grad_log_simulator(xi):
    """Gradient of log simulator N(mu, Sigma)."""
    return - solve(Sigma, xi)


def experiment(x00, T, N, epsilon, alphas, nlags):
    lam = epsilon
    log_pi = lambda xi: log_abc_posterior(xi, epsilon)
    ### COMMON VARIABLES
    v = q.rvs(N)
    log_uniforms1 = log(rand(N))     # Log uniforms for the HUG kernels
    log_uniforms2 = log(rand(N))     # Log uniforms for the HOP kernel
    u = MVN(zeros(d), eye(d)).rvs(N) # Original velocities for HOP kernel
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
    th_ess  = zeros((n_alphas, d))
    th_essj = zeros(n_alphas)
    th_rmse = zeros(n_alphas)
    th_uniq = zeros(n_alphas)
    th_act  = zeros((n_alphas, nlags))
    th_acu  = zeros((n_alphas, nlags))
    ### HUG + HOP
    x = x00
    for i in range(N):
        y, a1, e, eg, et = HugStepEJSD_Deterministic(x, v[i], log_uniforms1[i], T, B, q, log_pi, grad_log_simulator)
        x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, log_pi, grad_log_simulator)
        hh = vstack((hh, y, x))
        ahh1 += a1 * 100 / N
        ahh2 += a2 * 100 / N
        ehh += e / N
        eghh += eg / N 
        ethh += et / N 
    # COMPUTE ESS AND OTHER METRICS FOR HUG
    hh = hh[1:]
    hh_ess  = ESS_univariate(hh[::2])        # univariate ESS for each dimension
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
            y, a1, e, eg, et = HugTangentialStepEJSD_Deterministic(x, v[i], log_uniforms1[i], T, B, alpha, q, log_pi, grad_log_simulator)
            x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, log_pi, grad_log_simulator)
            th = vstack((th, y, x))
            ath1[k] += a1 * 100 / N
            ath2[k] += a2 * 100 / N
            eth[k]  += e / N
            egth[k] += eg / N 
            etth[k] += et / N 
        ### COMPUTE ESS AND OTHER METRISC FOR THUG
        th = th[1:]
        th_ess[k]  = ESS_univariate(th[::2])        # univariate ESS for each dimension
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
            'ESS': hh_ess,
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
            'ESS': th_ess,
            'ESS_J': th_essj,
            'RMSE': th_rmse,
            'UNIQUE': th_uniq,
            'AC_T': th_act,
            'AC_U': th_acu
        }
    }
    return out

        

# Target distribution is a diagonal MVN
Sigma = diag([1.0, 0.01, 1.0, 10.0, 1.0]) 
d = Sigma.shape[0]
target = MVN(zeros(d), Sigma)

# Proposal for velocity in HUG/THUG
q = MVN(zeros(d), eye(d))

# Sample a point at the beginning just to obtain some energy level
z0 = -10.650772399921479
# Function to sample a new point on the contour
new_point = lambda: GeneralizedEllipse(zeros(d), Sigma, exp(z0)).sample()

# Settings
B = 5 
N = 50000
kappa = 0.25    
n_runs = 50 #15
nlags = 2
n_cores = 8

Ts = [10, 1, 0.1, 0.01] 
epsilons = [0.000001]#[1.0, 0.001, 0.0001, 0.00001]
alphas = [0.8, 0.9, 0.95, 0.99, 0.995] 
n_epsilons = len(epsilons)
n_alphas = len(alphas)
n_T = len(Ts)

initial_time = time.time()
initial_points = vstack([new_point() for _ in range(n_runs)])
run_indices = np.arange(n_runs).tolist()  # One for each run
results = []

args = product(initial_points, [Ts], [N], [epsilons], [alphas], [nlags])

# FUNCTION TO RUN IN PARALLEL
def my_function(arguments):
    x00, Ts, N, epsilons, alphas, nlags = arguments
    out_dict = {'{}'.format(epsilon): {} for epsilon in epsilons}
    for epsilon in epsilons:
        for T in Ts:
            out_dict['{}'.format(epsilon)]['{}'.format(T)] = experiment(x00, T, N, epsilon, alphas, nlags)
    return out_dict


if __name__ == "__main__":
    # HUG
    ESS_HUG = zeros((n_runs, n_epsilons, n_T, d))           # Univariate ESS for each dim
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
    ESS_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas, d))      
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

    try:
        with Pool(n_cores) as p:
            results = p.map(my_function, args)
    except KeyboardInterrupt:
        p.terminate()
    except Exception as e:
        print('exception', e)
        p.terminate()
    finally:
        p.join()
    # Now I need to go through the results and store them 
    for i in range(n_runs):
        for j, epsilon in enumerate(epsilons):
            for k, T in enumerate(Ts):
                # Store HUG+HOP results
                ESS_HUG[i, j, k, :]    = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['ESS']
                ESS_JOINT_HUG[i, j, k] = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['ESS_J']
                A_HUG[i, j, k]         = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['A1']
                A_HOP_HUG[i, j, k]     = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['A2']
                RMSE_HUG[i, j, k]      = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['RMSE']
                EJSD_HUG[i, j, k]      = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['E']
                G_EJSD_HUG[i, j, k]    = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['EG']
                T_EJSD_HUG[i, j, k]    = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['ET']
                # Store THUG+HOP results
                ESS_THUG[i, j, k, :, :]      = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['ESS']
                ESS_JOINT_THUG[i, j, k, :]   = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['ESS_J']
                A_THUG[i, j, k, :]           = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['A1']
                A_HOP_THUG[i, j, k, :]       = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['A2']
                RMSE_THUG[i, j, k, :]        = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['RMSE']
                EJSD_THUG[i, j, k, :]        = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['E'] 
                G_EJSD_THUG[i, j, k, :]      = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['EG'] 
                T_EJSD_THUG[i, j, k, :]      = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['ET']

    print("Total time: ", time.time() - initial_time)

    # Save results
    folder = "experiment24e6/"

    save(folder + "EPSILONS.npy", epsilons)
    save(folder + "ALPHAS.npy", alphas)
    save(folder + "TS.npy", Ts)
    save(folder + "TIME.npy", np.array([time.time() - initial_time]))
    save(folder + "D.npy", d)

    save(folder + "ESS_HUG.npy", ESS_HUG)
    save(folder + "ESS_JOINT_HUG.npy", ESS_JOINT_HUG)
    save(folder + "A_HUG.npy", A_HUG)
    save(folder + "RMSE_HUG.npy", RMSE_HUG)
    save(folder + "EJSD_HUG.npy", EJSD_HUG)
    save(folder + "G_EJSD_HUG.npy", G_EJSD_HUG)
    save(folder + "T_EJSD_HUG.npy", T_EJSD_HUG)
    save(folder + "A_HOP_HUG.npy", A_HOP_HUG)

    save(folder + "ESS_THUG.npy", ESS_THUG)
    save(folder + "ESS_JOINT_THUG.npy", ESS_JOINT_THUG)
    save(folder + "A_THUG.npy", A_THUG)
    save(folder + "RMSE_THUG.npy", RMSE_THUG)
    save(folder + "EJSD_THUG.npy", EJSD_THUG)
    save(folder + "G_EJSD_THUG.npy", G_EJSD_THUG)
    save(folder + "T_EJSD_THUG.npy", T_EJSD_THUG)
    save(folder + "A_HOP_THUG.npy", A_HOP_THUG)

