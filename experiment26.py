# Experiment 26 - G and K model using HUG+HOP and THUG+HOP. This is run in parallel using Multiprocessing.

from multiprocessing import Process, Queue, Pool
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
    out = a_param + b_param*(1 + 0.8 * (1 - exp(-g_param * z)) / (1+exp(-g_param * z))) * ((1 + z**2)**k_param) * z
    return norm(out - y_star)

def data_generator(theta, N):
    """Generates initial observed data y_star."""
    z = randn(N)         # Get N samples from N(0, 1) for G&K simulation.
    a_param, b_param, k_param = theta   # Grab parameters
    return a_param + b_param*(1 + 0.8 * (1 - exp(-g_param * z)) / (1+exp(-g_param * z))) * ((1 + z**2)**k_param) * z

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
def experiment_thug(x00, T, N, alphas, epsilon, nlags):
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
    th_esst = zeros(n_alphas)
    th_essu = zeros(n_alphas)
    th_essj = zeros(n_alphas)
    th_uniq = zeros(n_alphas)
    th_act  = zeros((n_alphas, nlags))
    th_acu  = zeros((n_alphas, nlags))
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
    hh_esst = ESS_univariate(hh[::2, 0])     # ESS for theta
    hh_essu = ESS_univariate(hh[::2, 1])     # ESS for u
    hh_essj = ESS(hh[::2])                   # ESS joint
    hh_uniq = n_unique(hh)                             # Number of unique samples
    hh_act  = acf(hh[::2, 0], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for theta (remove the first 1.0)
    hh_acu  = acf(hh[::2, 1], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for u
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
        th_esst[k] = ESS_univariate(th[::2, 0])     # ESS for theta
        th_essu[k] = ESS_univariate(th[::2, 1])     # ESS for u
        th_essj[k] = ESS(th[::2])                   # ESS joint
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
            'UNIQUE': hh_uniq,
            'AC_T': hh_act,
            'AC_U': hh_acu,
            'SAMPLES': hh
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
            'UNIQUE': th_uniq,
            'AC_T': th_act,
            'AC_U': th_acu,
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
TS = [0.003]                         # Total integration time
B = 5                                # Number of bounces per iteration of HUG/THUG
N = 5000                              # Number of samples
epsilons = [0.1, 0.01]               # Tolerance for ABC
kappa = 0.001                        # HOP scaling in remaining directions relative to lam
nlags = 2                            # Number of lags to compute autocorrelation for
alphas = [0.5, 0.9]                  # Alphas for THUG
n_alphas = len(alphas)
n_runs = 8                           # Number of runs for each setting combination
g_param = 2.0                        # Parameter g is not estimated since it is un-informative
theta0 = np.array([3.0, 1.0, 0.5])   # True parameter for G and K model
y_star = data_generator(theta0, N=m) # Observed data

# HELPER FUNCTIONS
q = MVN(zeros(d), eye(d))                             # Spherically-symmetric proposal for HUG/THUG 
grad_function = grad(f)                               # Autograd gradient of deterministic simulator
func = lambda xi: np.r_[f(xi), zeros(d-1)]            # Function used to find initial point on manifold
guess = hstack((np.array([1.0, 1.0, 1.0]), zeros(m))) # Initial guess
xi0 = fsolve(func, guess)                             # Fsolve finds initiial point on manifold

def find_initial_points(n):
    initial_points = []
    while len(initial_points) < n:
        try:
            # Construct a new guess
            theta_guess = uniform(0.1, 3.0, size=3)
            guess = hstack((theta_guess, zeros(m)))
            # Solve to find point on manifold
            point = fsolve(func, guess)
            initial_points.append(point)
        except RuntimeWarning:  
            pass
    return vstack(initial_points)

# Initial points on manifold
initial_points = find_initial_points(n_runs)

# FUNCTION TO RUN IN PARALLEL
def my_function(run_index):
    out_dict = {'{}'.format(T): {} for T in TS}
    for T in TS:
        for epsilon in epsilons:
            out_dict['{}'.format(T)]['{}'.format(epsilon)] = experiment_thug(initial_points[run_index], T, N, alphas, epsilon, nlags)
    return out_dict


if __name__ == '__main__':
 
    # start_time = time.time()
    # queue = Queue()
    # processes = []

    # for run_index in range(n_runs):
    #     p = Process(target=my_function, args=(run_index, queue, ))
    #     processes.append(p)
    #     p.start()
    # print("processes run")
    # out = [queue.get() for _ in range(n_runs)]  # Now this containts the output.
    # print("queue gotten")


    # for process in processes:
    #     process.join()
    #     print("Jointed process: ", process)
    # print('processes joint')
    # print("TOTAL TIME: ", time.time() - start_time)

    start_time = time.time()
    run_indices = np.arange(n_runs).tolist()  # One for each run
    results = []
    try:
        with Pool(8) as p:
            results.append(p.map(my_function, run_indices))
    except KeyboardInterrupt:
        p.terminate()
    except Exception as e:
        p.terminate()
    finally:
        p.join()
    print("Total time: ", time.time() - start_time)
    

    
    