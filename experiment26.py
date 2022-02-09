# Experiment 26 - G and K model using HUG+HOP and THUG+HOP. This is run in parallel using Multiprocessing.
from multiprocessing import Pool
from itertools import product
## Autograd
import autograd.numpy as np
from autograd.numpy.random import seed, randn, rand
from autograd.numpy import exp, hstack, log
from autograd import grad, jacobian
from autograd.numpy.linalg import norm
from autograd.scipy.stats import norm as ag_norm
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f

# Standard Python Imports
from scipy.stats import norm as ndist
from scipy.stats import uniform as udist
from numpy import zeros, eye, vstack, sqrt, mean
from scipy.stats import multivariate_normal as MVN
from numpy.random import uniform
from scipy.optimize import fsolve
from statsmodels.tsa.stattools import acf
from numpy import save

# Custom functions
from tangential_hug_functions import HugTangentialStepEJSD_Deterministic, Hop_Deterministic, HugStepEJSD_Deterministic, HugRotatedStepEJSD_AR_Deterministic, HugRotatedStepEJSD_Deterministic
from utils import ESS_univariate, ESS, n_unique

import warnings
warnings.filterwarnings("error")

import time



def f(thetau):
    """Deterministic function for distance manifold. f:R^5 -> R """
    a_param, b_param, k_param, *z = thetau  # Latents are standard normal variables
    z = np.array(z)
    out = a_param + b_param*(1 + 0.8 * (1 - exp(-2.0 * z)) / (1+exp(-2.0 * z))) * ((1 + z**2)**k_param) * z
    return norm(out - y_star)

def data_generator(theta, N, rng_seed):
    """Generates initial observed data y_star."""
    seed(rng_seed)
    z = randn(N)         # Get N samples from N(0, 1) for G&K simulation.
    a_param, b_param, k_param = theta   # Grab parameters
    return a_param + b_param*(1 + 0.8 * (1 - exp(-2.0 * z)) / (1+exp(-2.0 * z))) * ((1 + z**2)**k_param) * z

def logprior(thetau):
    """Log prior distribution."""
    with np.errstate(divide='ignore'):
        return log((abs(thetau[:3]) <= 10).all().astype('float64')) + ndist.logpdf(thetau[3:]).sum()
    
def log_uniform_kernel(xi, epsilon):
    """Log density of uniform kernel. """
    with np.errstate(divide='ignore'):
        return log((f(xi) <= epsilon).astype('float64'))
    
def log_abc_posterior(xi, epsilon):
    """Log density of ABC posterior. Product of (param-latent) prior and uniform kernel."""
    return logprior(xi) + log_uniform_kernel(xi, epsilon)

#### thug
def experiment(x00, T, N, epsilon, alphas):
    """Runs Hug+Hop and THUG+HOP using the same velocities and the same random seeds.
    We also try to limit the noise in the HOP kernel by sampling the u variables beforehand.
    I run THUG for all values of alpha with the randomness fixed. 
    This is 1 run, for 1 epsilon. It does 1 HUG+HOP and then THUG+HOP for all alphas.
    T1: T for HUG
    T2: T for THUG
    """
    ### TARGET
    logpi = lambda xi: log_abc_posterior(xi, epsilon)
    lam = epsilon / 30
    d
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
    th_ess  = zeros((n_alphas, d))   ##### CHECK D
    th_essj = zeros(n_alphas)
    th_uniq = zeros(n_alphas)
    ### HUG + HOP
    x = x00
    for i in range(N):
        y, a1, e, eg, et = HugStepEJSD_Deterministic(x, v[i], log_uniforms1[i], T, B, q, logpi, grad_function)
        x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, logpi, grad_function)
        hh = vstack((hh, y, x))
        ahh1 += a1 * 100 / N
        ahh2 += a2 * 100 / N
        ehh += e / N
        eghh += eg / N 
        ethh += et / N 
    # COMPUTE ESS AND OTHER METRICS FOR HUG
    hh = hh[1:]
    hh_ess  = ESS_univariate(hh[::2])         # univariate ESS for each dimension
    hh_essj = ESS(hh[::2])                   # ESS joint
    hh_uniq = n_unique(hh)                             # Number of unique samples
    ### THUG + HOP
    for k, alpha in enumerate(alphas):
        x = x00
        th = x00      # RESTART THE SAMPLES FROM SCRATCH
        for i in range(N):
            y, a1, e, eg, et = HugTangentialStepEJSD_Deterministic(x, v[i], log_uniforms1[i], T, B, alpha, q, logpi, grad_function)
            x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, logpi, grad_function)
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
        th_uniq[k] = n_unique(th)                             # Number of unique samples
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
            'UNIQUE': hh_uniq,
            'SAMPLES': hh
        },
        'TH': {
            'A1': ath1,
            'A2': ath2,
            'E': eth,
            'EG': egth, 
            'ET': etth, 
            'ESS': th_ess,
            'ESS_J': th_essj,
            'UNIQUE': th_uniq,
            'SAMPLES': th
        },
        'N': N,
        'T': T,
        'ALPHAS': alphas,
        'EPSILON': epsilon
    }
    return out


# SETTINGS
m = 2                                # Number of latents in G and K model
d = 3 + m                            # Dimension of ambient space (theta has dim 3)
Ts = [10.0, 1.0, 0.1, 0.01]                         # Total integration time
B = 5                                # Number of bounces per iteration of HUG/THUG
N = 20000                            # Number of samples
epsilons = [1.0, 0.1, 0.01]          # Tolerance for ABC
kappa = 0.005                        # HOP scaling in remaining directions relative to lam
nlags = 2                            # Number of lags to compute autocorrelation for
alphas = [0.5, 0.9, 0.95]                  # Alphas for THUG
n_alphas = len(alphas)
n_cores = 8
n_epsilons = len(epsilons)
n_T = len(Ts)
n_runs = 16                          # Number of runs for each setting combination
rng_seed = 1234
theta0 = np.array([3.0, 1.0, 0.5])   # True parameter for G and K model
y_star = data_generator(theta0, N=m, rng_seed=rng_seed) # Observed data

# HELPER FUNCTIONS
q = MVN(zeros(d), eye(d))                             # Spherically-symmetric proposal for HUG/THUG 
grad_function = grad(f)                               # Autograd gradient of deterministic simulator
func = lambda xi: np.r_[f(xi), zeros(d-1)]            # Function used to find initial point on manifold


def find_initial_points(n):
    initial_points = []
    while len(initial_points) < n:
        try:
            # Construct a new guess
            theta_guess = uniform(0.1, 3.0, size=3)
            guess = hstack((theta_guess, zeros(m)))
            # Solve to find point on manifold
            point = fsolve(func, guess)
            if not np.isfinite([log_abc_posterior(point, epsilons[i]) for i in range(n_epsilons)]).all():
                pass
            else:
                initial_points.append(point)
        except RuntimeWarning:
            pass
    return vstack(initial_points)

# Initial points on manifold
initial_time = time.time()
initial_points = find_initial_points(n_runs)
run_indices = np.arange(n_runs).tolist() 
args = list(product(initial_points, [Ts], [N], [epsilons], [alphas]))

# FUNCTION TO RUN IN PARALLEL
def my_function(arguments):
    x00, Ts, N, epsilons, alphas = arguments
    out_dict = {'{}'.format(epsilon): {} for epsilon in epsilons}
    for epsilon in epsilons:
        for T in Ts:
            out_dict['{}'.format(epsilon)]['{}'.format(T)] = experiment(x00, T, N, epsilon, alphas)
    return out_dict


if __name__ == '__main__':
    # HUG
    ESS_HUG = zeros((n_runs, n_epsilons, n_T, d))           # Univariate ESS for each dim
    ESS_JOINT_HUG = zeros((n_runs, n_epsilons, n_T))        # multiESS on whole chain
    A_HUG = zeros((n_runs, n_epsilons, n_T))                # Acceptance probability
    RMSE_HUG = zeros((n_runs, n_epsilons, n_T))             # Root Mean Squared Error
    EJSD_HUG = zeros((n_runs, n_epsilons, n_T))             # Full EJSD 
    G_EJSD_HUG = zeros((n_runs, n_epsilons, n_T))           # EJSD for gradient only
    T_EJSD_HUG = zeros((n_runs, n_epsilons, n_T))           # EJSD for tangent only
    A_HOP_HUG  = zeros((n_runs, n_epsilons, n_T))           # Acceptance probability of HOP for HUG.

    # THUG
    ESS_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas, d))      
    ESS_JOINT_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))        
    A_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))            
    RMSE_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))    
    EJSD_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))             
    G_EJSD_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))           
    T_EJSD_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))
    A_HOP_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas)) 

    try:
        with Pool(n_cores) as p:
            results = p.map(my_function, args)
    except KeyboardInterrupt:
        p.terminate()
    except Exception as e:
        print('Exception-Pool:', e)
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
                EJSD_HUG[i, j, k]      = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['E']
                G_EJSD_HUG[i, j, k]    = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['EG']
                T_EJSD_HUG[i, j, k]    = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['ET']
                # Store THUG+HOP results
                ESS_THUG[i, j, k, :, :]      = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['ESS']
                ESS_JOINT_THUG[i, j, k, :]   = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['ESS_J']
                A_THUG[i, j, k, :]           = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['A1']
                A_HOP_THUG[i, j, k, :]       = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['A2']
                EJSD_THUG[i, j, k, :]        = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['E'] 
                G_EJSD_THUG[i, j, k, :]      = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['EG'] 
                T_EJSD_THUG[i, j, k, :]      = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['ET']

    print("Total time: ", time.time() - initial_time)

    # Save results
    folder = "experiment26/"

    save(folder + "EPSILONS.npy", epsilons)
    save(folder + "ALPHAS.npy", alphas)
    save(folder + "TS.npy", Ts)
    save(folder + "TIME.npy", np.array([time.time() - initial_time]))
    save(folder + "D.npy", d)

    save(folder + "ESS_HUG.npy", ESS_HUG)
    save(folder + "ESS_JOINT_HUG.npy", ESS_JOINT_HUG)
    save(folder + "A_HUG.npy", A_HUG)
    save(folder + "EJSD_HUG.npy", EJSD_HUG)
    save(folder + "G_EJSD_HUG.npy", G_EJSD_HUG)
    save(folder + "T_EJSD_HUG.npy", T_EJSD_HUG)
    save(folder + "A_HOP_HUG.npy", A_HOP_HUG)

    save(folder + "ESS_THUG.npy", ESS_THUG)
    save(folder + "ESS_JOINT_THUG.npy", ESS_JOINT_THUG)
    save(folder + "A_THUG.npy", A_THUG)
    save(folder + "EJSD_THUG.npy", EJSD_THUG)
    save(folder + "G_EJSD_THUG.npy", G_EJSD_THUG)
    save(folder + "T_EJSD_THUG.npy", T_EJSD_THUG)
    save(folder + "A_HOP_THUG.npy", A_HOP_THUG)



    
    