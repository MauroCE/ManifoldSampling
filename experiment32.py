# Experiment 32: HUG+HOP vs THUG+HOP. G and K distribution.
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save, exp, linspace, pi
from numpy.linalg import solve, norm
from scipy.stats import multivariate_normal as MVN
from scipy.stats.stats import sigmaclip
from scipy.optimize import fsolve
from tangential_hug_functions import HugTangentialStepEJSD_Deterministic, Hop_Deterministic, HugStepEJSD_Deterministic
from utils import ESS_univariate, ESS, n_unique
from numpy.random import normal, rand, uniform, randn
from statsmodels.tsa.stattools import acf
from Manifolds.RotatedEllipse import RotatedEllipse
import time
from warnings import catch_warnings, filterwarnings
from scipy.stats import norm as ndist
from multiprocessing import Pool
from itertools import product


def f(xi): 
    """Simulator. This is a deterministic function."""
    a, b, g, k, *z = xi
    z = np.array(z)
    return a + b*(1 + 0.8 * (1 - exp(-g * z)) / (1 + exp(-g * z))) * ((1 + z**2)**k) * z

def f_broadcast(xi_matrix):
    """Broadcasted version of f."""
    return np.apply_along_axis(f, 1, xi_matrix)

def fnorm(xi, ystar):
    """This function is h(xi) = |f(xi) - y*|. Basically the function defining Chang's manifold."""
    return norm(f(xi) - ystar)

def fnorm_broadcast(xi_matrix, ystar):
    """Broadcasted version of fnorm."""
    return norm(f_broadcast(xi_matrix) - ystar, axis=1)

def Jf_transpose(xi):
    """Jacobian function of f."""
    _, b, g, k, *z = xi
    z = np.array(z)
    return np.vstack((
        np.ones(len(z)),
        (1 + 0.8 * (1 - exp(-g * z)) / (1 + exp(-g * z))) * ((1 + z**2)**k) * z,
        8 * b * (z**2) * ((1 + z**2)**k) * exp(g*z) / (5 * (1 + exp(g*z))**2),
        b*z*((1+z**2)**k)*(1 + 9*exp(g*z))*log(1 + z**2) / (5*(1 + exp(g*z))),
        np.diag(b*((1+z**2)**(k-1))*(((18*k + 9)*(z**2) + 9)*exp(2*g*z) + (8*g*z**3 + (20*k + 10)*z**2 + 8*g*z + 10)*exp(g*z) + (2*k + 1)*z**2 + 1) / (5*(1 + exp(g*z))**2))
    ))

def grad_fnorm(xi, ystar):
    """Gradient of h(xi)."""
    return Jf_transpose(xi) @ (f(xi) - ystar)

def logprior(xi):
    theta, z = xi[:4], xi[4:]
    with np.errstate(divide='ignore'):
        return log(((0 <= theta) & (theta <= 10)).all().astype('float64')) + ndist.logpdf(z).sum()

def logprior_broadcast(xi_matrix):
    with np.errstate(divide='ignore'):
        return ((0 <= xi_matrix[:, :4]) & (xi_matrix[:, :4] <= 10)).all(axis=1).astype('float64') + ndist.logpdf(xi_matrix[:, 4:]).sum(axis=1)

def sample_prior(n_params=4, n_latents=20):
    """Sample from prior distribution over params and latents."""
    return np.r_[uniform(low=0.0, high=10.0, size=n_params), randn(n_latents)]

# def logprior_broadcast(xi_matrix):
#     """Same as logprior but broadcasted to a whole matrix."""
#     nz = xi_matrix.shape[1] - 1
#     return -log(10) + MVN(zeros(nz), eye(nz)).logpdf(xi_matrix[:, 1:])

def log_epanechnikov_kernel(xi, epsilon, fnorm, ystar):
    u = fnorm(xi, ystar)
    with np.errstate(divide='ignore'):
        return log((3*(1 - (u**2 / (epsilon**2))) / (4*epsilon)) * float(u <= epsilon))

def log_epanechnikov_kernel_broadcast(xi_matrix, epsilon, fnorm_broadcast, ystar):
    u_vector = fnorm_broadcast(xi_matrix, ystar)
    with np.errstate(divide='ignore'):
        return log((3*(1 - (u_vector**2 / (epsilon**2))) / (4*epsilon)) * (u_vector <= epsilon).astype('float'))

# def log_epanechnikov_kernel_broadcast(xi_matrix, epsilon, fnorm_broadcast, y_star):
#     """Same as log_epanechnikov kernel but broadcasted to a matrix."""
#     u = fnorm_broadcast(xi_matrix, y_star)
#     with np.errstate(divide='ignore'):
#         return log((3*(1 - (u**2 / (epsilon**2))) / (4*epsilon)) * (u <= epsilon).astype('float')) 

# def log_abc_posterior(xi, epsilon):
#     """Log density of ABC posterior. Product of (param-latent) prior and Epanechnikov kernel."""
#     return logprior(xi) + log_epanechnikov_kernel(xi, epsilon)

# def log_abc_posterior_broadcast(xi_matrix, epsilon):
#     """Same as log_abc_posterior but broadcasted to a matrix."""
#     return logprior_broadcast(xi_matrix) + log_epanechnikov_kernel_broadcast(xi_matrix, epsilon)

def data_generator(theta, N, seed):
    """Generates data with a given random seed."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=N)
    return f(np.r_[theta, z])

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


def experiment(x00, T, B, N, kappa, epsilon, alphas, q, fnorm, y_star):
    """Runs Hug+Hop and THUG+HOP using the same velocities and the same random seeds.
    We also try to limit the noise in the HOP kernel by sampling the u variables beforehand.
    I run THUG for all values of alpha with the randomness fixed. 
    This is 1 run, for 1 epsilon. It does 1 HUG+HOP and then THUG+HOP for all alphas.
    T1: T for HUG
    T2: T for THUG
    """
    lam = epsilon
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
    # dkth = zeros((n_alphas, N))  # delta k
    # dlth = zeros((n_alphas, N))  # delta l
    # afth = zeros((n_alphas, N))  # acceptance flag for THUG
    ### ADDITIONAL STORAGE FOR THUG
    th_ess = zeros((n_alphas, d))
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
    hh_ess = ESS_univariate(hh[::2])
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
            y, a1, e, eg, et = HugTangentialStepEJSD_Deterministic(x, v[i], log_uniforms1[i], T, B, alpha, q, log_post, grad_log_sim)
            x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, log_post, grad_log_sim)
            th = vstack((th, y, x))
            # dkth[k, i] = dk 
            # dlth[k, i] = dl
            # afth[k, i] = a1
            ath1[k] += a1 * 100 / N
            ath2[k] += a2 * 100 / N
            eth[k]  += e / N
            egth[k] += eg / N 
            etth[k] += et / N 
        th_time[k] = time.time() - th_start
        ### COMPUTE ESS AND OTHER METRISC FOR THUG
        th = th[1:]
        th_ess[k] = check(ESS_univariate(th[::2]), "univESS TH")
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
            'ESS': hh_ess,
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
            'E': eth,
            'EG': egth, 
            'ET': etth, 
            'ESS': th_ess,
            'ESS_J': th_essj,
            'ESS_LOGPI': th_ess_logpi,
            'ESS_J_TOT': th_essjtot,
            'RMSE': th_rmse,
            'UNIQUE': th_uniq,
            'SAMPLES': th,
            'TIME': th_time
        },
        'N': N,
        'T': T,
        'ALPHAS': alphas,
        'EPSILON': epsilon
    }
    return out


# Need to find a point near the manifold to start with
def new_point(n_params, n_latents, fnorm, y_star, max_iter, epsilons, threshold):
    func = lambda xi: np.r_[fnorm(xi, y_star), zeros(n_params + n_latents-1)]
    log_abc_posterior = lambda xi, epsilon: logprior(xi) + log_epanechnikov_kernel(xi, epsilon, fnorm, y_star)
    i = 0
    n_epsilons = len(epsilons)
    with catch_warnings():
        filterwarnings('error')
        while i <= max_iter:
            i += 1
            try: 
                theta_guess = uniform(0.0, 10.0, size=4)
                guess = np.hstack((theta_guess, zeros(n_latents)))
                point = fsolve(func, guess)
                if not np.isfinite([log_abc_posterior(point, epsilons[i]) for i in range(n_epsilons)]).all():
                    pass
                else:
                    if fnorm(point, y_star) < threshold:
                        return point
                    else:
                        continue
            except RuntimeWarning:
                continue



initial_time = time.time()
# Generate observed data
n_latents = 10
seed = 1234
theta0 = np.array([3.0, 1.0, 2.0, 0.5])
n_params = len(theta0)
y_star = data_generator(theta0, n_latents, seed)

# Total dimensionality of \xi space is nz+1
d = n_latents + n_params

# Proposal for velocity in HUG/THUG
q = MVN(zeros(d), eye(d)) 
# Settings
B = 5 
N = 10000
kappa = 0.25    
n_runs = 20 #15
nlags = 20
n_cores = 8

max_iter_newpoint = 2000
threshold_newpoint = 1e-5

Ts = [1.0, 0.1, 0.01] #[0.5, 0.3, 0.1] #[10, 1, 0.1] #[7, 5, 3, 1, 0.1, 0.01]
epsilons = [10.0, 1.0, 0.5] #[0.1, 0.0001] #[0.1, 0.001, 0.00001, 0.0000001]
alphas = [0.8, 0.9] #[0.6, 0.7, 0.8, 0.9, 0.95]
n_epsilons = len(epsilons)
n_alphas = len(alphas)
n_T = len(Ts)

# Initial points
initial_points = np.vstack([new_point(n_params, n_latents, fnorm, y_star, max_iter_newpoint, epsilons, threshold_newpoint) for _ in range(n_runs)])
run_indices = np.arange(n_runs).tolist() 
args = list(product(initial_points, [Ts], [B], [N], [kappa], [epsilons], [alphas], [q], [fnorm], [y_star]))

# Function to run in parallel
def my_function(arguments):
    x00, Ts, B, N, kappa, epsilons, alphas, q, fnorm, y_star  = arguments
    out_dict = {'{}'.format(epsilon): {} for epsilon in epsilons}
    for epsilon in epsilons:
        for T in Ts:
            out_dict['{}'.format(epsilon)]['{}'.format(T)] = experiment(x00, T, B, N, kappa, epsilon, alphas, q, fnorm, y_star)
    return out_dict



if __name__ == "__main__":
    # HUG
    ESS_HUG = zeros((n_runs, n_epsilons, n_T, d))           # Univariate ESS for each dimension
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
    ESS_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas, d))      
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
    TIME_THUG = zeros((n_runs, n_epsilons, n_T, n_alphas))

    # Run things in parallel
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

    # Now go through the results and store them
    for i in range(n_runs):
        for j, epsilon in enumerate(epsilons):
            for k, T in enumerate(Ts):
                # Store HUG+HOP results
                ESS_HUG[i, j, k, :]        = check(results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['ESS'], "uniESS HH")
                LOGPI_ESS_HUG[i, j, k]     = check(results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['ESS_LOGPI'], "LOGPI ESS HH")
                ESS_JOINT_HUG[i, j, k]     = check(results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['ESS_J'], "ESS HH")
                ESS_JOINT_TOT_HUG[i, j, k] = check(results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['ESS_J_TOT'], "ESS HH TOT")
                A_HUG[i, j, k]             = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['A1']
                A_HOP_HUG[i, j, k]         = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['A2']
                EJSD_HUG[i, j, k]          = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['E']
                G_EJSD_HUG[i, j, k]        = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['EG']
                T_EJSD_HUG[i, j, k]        = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['ET']
                N_UNIQUE_HUG[i, j, k]      = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']['UNIQUE']
                TIME_HUG[i, j, k]          = results[i]['{}'.format(epsilon)]['{}'.format(T)]['HH']["TIME"]
                # Store THUG+HOP results
                ESS_THUG[i, j, k, :, :]        = check(results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['ESS'], "univESS TH")
                LOGPI_ESS_THUG[i, j, k, :]     = check(results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['ESS_LOGPI'], "LOGPI ESS TH")
                ESS_JOINT_THUG[i, j, k, :]     = check(results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['ESS_J'], "ESS TH")
                ESS_JOINT_TOT_THUG[i, j, k, :] = check(results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['ESS_J_TOT'], "ESS TH TOT")
                A_THUG[i, j, k, :]             = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['A1']
                A_HOP_THUG[i, j, k, :]         = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['A2']
                EJSD_THUG[i, j, k, :]          = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['E'] 
                G_EJSD_THUG[i, j, k, :]        = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['EG'] 
                T_EJSD_THUG[i, j, k, :]        = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['ET']
                N_UNIQUE_THUG[i, j, k, :]      = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['UNIQUE']
                TIME_THUG[i, j, k, :]          = results[i]['{}'.format(epsilon)]['{}'.format(T)]['TH']['TIME']

 

    print("Total time: ", time.time() - initial_time)

    # for i in range(n_runs):
    #     print("Run: ", i)
    #     # We need a new point for each run, but then must be the same for all other settings
    #     initial_point = new_point(n_params, n_latents, fnorm, y_star, max_iter_newpoint, epsilons, threshold_newpoint)
    #     for j, epsilon in enumerate(epsilons):
    #         lam = epsilon # For HOP
    #         for k, T in enumerate(Ts):
    #             out = experiment(initial_point, T, B, N, kappa, epsilon, alphas, q, fnorm, y_star)

    #             # Store HUG results
    #             ESS_HUG[i, j, k, :]      = check(out['HH']['ESS'], "univESS HH")
    #             LOGPI_ESS_HUG[i, j, k]   = check(out['HH']['ESS_LOGPI'], "LOGPI ESS HH")
    #             ESS_JOINT_HUG[i, j, k]   = check(out['HH']['ESS_J'], "ESS HH")
    #             ESS_JOINT_TOT_HUG[i, j, k] = check(out['HH']['ESS_J_TOT'], "ESS HH TOT")
    #             A_HUG[i, j, k]           = out['HH']['A1']
    #             A_HOP_HUG[i, j, k]       = out['HH']['A2']
    #             RMSE_HUG[i, j, k]        = out['HH']['RMSE']
    #             EJSD_HUG[i, j, k]        = out['HH']['E']
    #             G_EJSD_HUG[i, j, k]      = out['HH']['EG']
    #             T_EJSD_HUG[i, j, k]      = out['HH']['ET']
    #             N_UNIQUE_HUG[i, j, k]    = out['HH']['UNIQUE']
    #             TIME_HUG[i, j, k]        = out['HH']["TIME"]

    #             # Store THUG results
    #             ESS_THUG[i, j, k, :, :]      = check(out['TH']['ESS'], "univESS TH")
    #             LOGPI_ESS_THUG[i, j, k, :]   = check(out['TH']['ESS_LOGPI'], "LOGPI ESS TH")
    #             ESS_JOINT_THUG[i, j, k, :]   = check(out['TH']['ESS_J'], "ESS TH")
    #             ESS_JOINT_TOT_THUG[i, j, k, :] = out['TH']['ESS_J_TOT']
    #             A_THUG[i, j, k, :]           = out['TH']['A1']
    #             A_HOP_THUG[i, j, k, :]       = out['TH']['A2']
    #             RMSE_THUG[i, j, k, :]        = out['TH']['RMSE']
    #             EJSD_THUG[i, j, k, :]        = out['TH']['E'] 
    #             G_EJSD_THUG[i, j, k, :]      = out['TH']['EG'] 
    #             T_EJSD_THUG[i, j, k, :]      = out['TH']['ET']
    #             N_UNIQUE_THUG[i, j, k, :]    = out['TH']['UNIQUE']
    #             TIME_THUG[i, j, k, :]        = out['TH']['TIME']
                

    # Save results
    folder = "dumper4/" #"experiment32/" 

    save(folder + "EPSILONS.npy", epsilons)
    save(folder + "ALPHAS.npy", alphas)
    save(folder + "TS.npy", Ts)
    save(folder + "TIME.npy", np.array([time.time() - initial_time]))
    save(folder + "N.npy", np.array([N]))
    save(folder + "D.npy", d)

    save(folder + "ESS_HUG.npy", ESS_HUG)
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

    save(folder + "ESS_THUG.npy", ESS_THUG)
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
    save(folder + "TIME_THUG.npy", TIME_THUG)

