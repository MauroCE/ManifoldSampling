# Experiment 20: THUG + HOP vs THUG-AR + HOP in LFI setting with uniform kernel and uniform prior.
# We use the degenerate AR process. We focus on alpha=0.99. 
import numpy as np
from numpy import zeros, diag, eye, log, sqrt, vstack, mean, save
from numpy.core.fromnumeric import take
from numpy.linalg import solve
from scipy.stats import multivariate_normal as MVN
from tangential_hug_functions import Hop_Deterministic
from tangential_hug_functions import HugTangentialStepEJSD_Deterministic
from tangential_hug_functions import HugTangentialStepEJSD_AR_Deterministic
from utils import ESS_univariate, ESS, n_unique
from numpy.random import normal, uniform, rand
from statsmodels.tsa.stattools import acf


def log_uniform_kernel(xi, epsilon):
    """Log density of uniform kernel. """
    with np.errstate(divide='ignore'):
        return log((abs(target.logpdf(xi) - z0) <= epsilon).astype('float64'))
    
def logprior_uniform(xi):
    """Log density for uniform prior p(xi) of parameters and latents U([-5,5]x[-5,5])."""
    with np.errstate(divide='ignore'):
        return log((abs(xi) <= 5.0).all().astype('float64'))

def logprior_uniform_all(xi):
    """Log density for uniform prior p(xi) of parameters and latents U([-5,5]x[-5,5])."""
    with np.errstate(divide='ignore'):
        return log((abs(xi) <= 5.0).all(axis=1).astype('float64'))
    
def log_abc_posterior(xi):
    """Log density of ABC posterior. Product of (param-latent) prior and uniform kernel."""
    return logprior_uniform(xi) + log_uniform_kernel(xi, epsilon)

def log_abc_posterior_all(xi):
    """Log density of ABC posterior. Product of (param-latent) prior and uniform kernel."""
    return logprior_uniform_all(xi) + log_uniform_kernel(xi, epsilon)
    
def grad_log_simulator(xi):
    """Gradient of log simulator N(mu, Sigma)."""
    return - solve(Sigma, xi)




def experiment(x00, N, alpha, probs, nlags):
    """Runs Thug+Hop and Thug-AR+HOP using the same velocities and the same random seeds. 
    Randomness is shared. Basically quite similar to experiment 13 but now we see if the 
    AR process aids THUG or not. To make things easier, I run it for a single alpha. 
    I will change alpha based on the particular run of the experiment. 
    """
    ### COMMON VARIABLES
    v = q.rvs(N)
    log_uniforms1 = log(rand(N))     # Log uniforms for the THUG kernels
    log_uniforms2 = log(rand(N))     # Log uniforms for the HOP kernel
    u = MVN(zeros(2), eye(2)).rvs(N) # Original velocities for HOP kernel
    ### STORAGE THUG + HOP (will be denoted TH)
    th = x00              # Initial sample
    ath1 = 0.0            # Acceptance probability for THUG kernel
    ath2 = 0.0            # Acceptance probability for HOP kernel (when used with THUG)
    eth = 0.0             # EJSD
    egth = 0.0            # EJSD in Gradient direction
    etth = 0.0            # EJSD in Tangent direction
    ### STORAGE THUG-AR + HOP (will be denoted TA)
    ata1 = zeros(n_probs)
    ata2 = zeros(n_probs)
    eta  = zeros(n_probs)
    egta = zeros(n_probs)
    etta = zeros(n_probs)
    ### ADDITIONAL STORAGE FOR THUG-AR + HOP
    ta_esst = zeros(n_probs)
    ta_essu = zeros(n_probs)
    ta_essj = zeros(n_probs)
    ta_rmse = zeros(n_probs)
    ta_uniq = zeros(n_probs)
    ta_act  = zeros((n_probs, nlags))
    ta_acu  = zeros((n_probs, nlags))
    ### THUG + HOP
    x = x00
    for i in range(N):
        y, a1, e, eg, et = HugTangentialStepEJSD_Deterministic(x, v[i], log_uniforms1[i], T, B, alpha, q, log_abc_posterior, grad_log_simulator)
        x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, log_abc_posterior, grad_log_simulator)
        th = vstack((th, y, x))
        ath1 += a1 * 100 / N
        ath2 += a2 * 100 / N
        eth  += e / N
        egth += eg / N 
        etth += et / N 
    # COMPUTE ESS AND OTHER METRICS FOR THUG + HOP.
    th = th[1:]
    th_esst = ESS_univariate(th[:, 0])     # ESS for theta
    th_essu = ESS_univariate(th[:, 1])     # ESS for u
    th_essj = ESS(th)                      # ESS joint
    th_rmse = sqrt(mean((target.logpdf(th) - z0)**2))  # RMSE on energy
    th_uniq = n_unique(th)                             # Number of unique samples
    th_act  = acf(th[:, 0], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for theta (remove the first 1.0)
    th_acu  = acf(th[:, 1], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for u
    ### THUG-AR + HOP
    for k, prob in enumerate(probs):
        x = x00
        ta = x00     # RESTART THE SAMPLES
        velocity = v[0]
        for i in range(N):
            if uniform() > prob:  # AR PROCESS
                velocity = v[i]
            y, velocity, a1, e, eg, et = HugTangentialStepEJSD_AR_Deterministic(x, velocity, log_uniforms1[i], T, B, alpha, q, log_abc_posterior, grad_log_simulator)
            x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, log_abc_posterior, grad_log_simulator)
            ta = vstack((ta, y, x))
            ata1[k] += a1 * 100 / N
            ata2[k] += a2 * 100 / N
            eta[k]  += e / N
            egta[k] += eg / N 
            etta[k] += et / N 
        ### COMPUTE ESS AND OTHER METRISC FOR THUG-AR
        ta = ta[1:]
        ta_esst[k] = ESS_univariate(ta[:, 0])     # ESS for theta
        ta_essu[k] = ESS_univariate(ta[:, 1])     # ESS for u
        ta_essj[k] = ESS(ta)                      # ESS joint
        ta_rmse[k] = sqrt(mean((target.logpdf(ta) - z0)**2))  # RMSE on energy
        ta_uniq[k] = n_unique(ta)                             # Number of unique samples
        ta_act[k] = acf(ta[:, 0], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for theta
        ta_acu[k] = acf(ta[:, 1], adjusted=True, nlags=nlags, fft=True)[1:]  # Autocorrelation for u
    # RETURN EVERYTHING
    out = {
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
        },
        'TA': {
            'A1': ata1,
            'A2': ata2,
            'E': eta,
            'EG': egta, 
            'ET': etta, 
            'ESS_T': ta_esst,
            'ESS_U': ta_essu,
            'ESS_J': ta_essj,
            'RMSE': ta_rmse,
            'UNIQUE': ta_uniq,
            'AC_T': ta_act,
            'AC_U': ta_acu
        }
    }
    return out

   
if __name__ == "__main__":
    # Target distribution is a diagonal MVN
    Sigma0 = diag([1.0, 5.0])
    coef = 1.0
    Sigma = coef * Sigma0
    target = MVN(zeros(2), Sigma)

    # Proposal for velocity in HUG/HUG_AR
    q = MVN(zeros(2), eye(2))


    # Settings
    T = 1.5
    B = 5
    N = 20000
    kappa = 0.25
    n_runs = 10
    alpha = 0.9
    nlags = 10

    epsilons = [0.1, 0.001, 0.00001, 0.0000001]
    probs = [0.5, 0.75, 0.95]

    n_epsilons = len(epsilons)
    n_probs = len(probs)

    # THUG + HOP
    THETA_ESS_THUG = zeros((n_runs, n_epsilons))        # Univariate ESS for \theta chain
    U_ESS_THUG = zeros((n_runs, n_epsilons))            # Univariate ESS for u chain
    ESS_THUG = zeros((n_runs, n_epsilons))              # multiESS on whole chain
    A_THUG = zeros((n_runs, n_epsilons))                # Acceptance probability
    RMSE_THUG = zeros((n_runs, n_epsilons))             # Root Mean Squared Error
    EJSD_THUG = zeros((n_runs, n_epsilons))             # Full EJSD 
    G_EJSD_THUG = zeros((n_runs, n_epsilons))           # EJSD for gradient only
    T_EJSD_THUG = zeros((n_runs, n_epsilons))           # EJSD for tangent only
    A_HOP_THUG  = zeros((n_runs, n_epsilons))           # Acceptance probability of HOP for THUG.
    N_UNIQUE_THUG = zeros((n_runs, n_epsilons))         # Number of unique samples
    THETA_AC_THUG = zeros((n_runs, n_epsilons, nlags))  # Autocorrelation for theta
    U_AC_THUG = zeros((n_runs, n_epsilons, nlags))      # Autocorrelation for u

    # THUG-AR + HOP
    THETA_ESS_THUG_AR = zeros((n_runs, n_epsilons, n_probs))      
    U_ESS_THUG_AR = zeros((n_runs, n_epsilons, n_probs))  
    ESS_THUG_AR = zeros((n_runs, n_epsilons, n_probs))        
    A_THUG_AR = zeros((n_runs, n_epsilons, n_probs))            
    RMSE_THUG_AR = zeros((n_runs, n_epsilons, n_probs))    
    EJSD_THUG_AR = zeros((n_runs, n_epsilons, n_probs))             
    G_EJSD_THUG_AR = zeros((n_runs, n_epsilons, n_probs))           
    T_EJSD_THUG_AR = zeros((n_runs, n_epsilons, n_probs))
    A_HOP_THUG_AR = zeros((n_runs, n_epsilons, n_probs))    
    N_UNIQUE_THUG_AR = zeros((n_runs, n_epsilons, n_probs))        
    THETA_AC_THUG_AR = zeros((n_runs, n_epsilons, n_probs, nlags)) 
    U_AC_THUG_AR = zeros((n_runs, n_epsilons, n_probs, nlags)) 

    for j, epsilon in enumerate(epsilons):
        lam = epsilon  
        for i in range(n_runs):
            # GET NEW STARTING POINT
            x0 = normal(size=2)                           # Keep initial point the same
            z0 = target.logpdf(x0)                        # Feed through simulator
            # RUN EXPERIMENT
            out = experiment(x0, N, alpha, probs, nlags)
            # STORE THUG+HOP RESULTS
            THETA_ESS_THUG[i, j]   = out['TH']['ESS_T']
            U_ESS_THUG[i, j]       = out['TH']['ESS_U']
            ESS_THUG[i, j]         = out['TH']['ESS_J']
            A_THUG[i, j]           = out['TH']['A1']
            A_HOP_THUG[i, j]       = out['TH']['A2']
            RMSE_THUG[i, j]        = out['TH']['RMSE']
            EJSD_THUG[i, j]        = out['TH']['E']
            G_EJSD_THUG[i, j]      = out['TH']['EG']
            T_EJSD_THUG[i, j]      = out['TH']['ET']
            N_UNIQUE_THUG[i, j]    = out['TH']['UNIQUE']
            THETA_AC_THUG[i, j, :] = out['TH']['AC_T']
            U_AC_THUG[i, j, :]     = out['TH']['AC_U']
            # STORE THUG-AR+HOP RESULTS
            THETA_ESS_THUG_AR[i, j, :]   = out['TA']['ESS_T']
            U_ESS_THUG_AR[i, j, :]       = out['TA']['ESS_U']
            ESS_THUG_AR[i, j, :]         = out['TA']['ESS_J']
            A_THUG_AR[i, j, :]           = out['TA']['A1']
            A_HOP_THUG_AR[i, j, :]       = out['TA']['A2']
            RMSE_THUG_AR[i, j, :]        = out['TA']['RMSE']
            EJSD_THUG_AR[i, j, :]        = out['TA']['E'] 
            G_EJSD_THUG_AR[i, j, :]      = out['TA']['EG'] 
            T_EJSD_THUG_AR[i, j, :]      = out['TA']['ET']
            N_UNIQUE_THUG_AR[i, j, :]    = out['TA']['UNIQUE']
            THETA_AC_THUG_AR[i, j, :, :] = out['TA']['AC_T']
            U_AC_THUG_AR[i, j, :, :]     = out['TA']['AC_U']

    # IMPORTANTLY, THERE WILL BE A FOLDER FOR EACH DIFFERENT ALPHA
    folder = "experiment20/" + str(alpha).replace('.', '') + "/"

    save(folder + "EPSILONS.npy", epsilons)
    save(folder + "PROBS.npy", probs)
    save(folder + "ALPHA.npy", alpha)

    save(folder + "THETA_ESS_THUG.npy", THETA_ESS_THUG)
    save(folder + "U_ESS_THUG.npy", U_ESS_THUG)
    save(folder + "ESS_THUG.npy", ESS_THUG)
    save(folder + "A_THUG.npy", A_THUG)
    save(folder + "RMSE_THUG.npy", RMSE_THUG)
    save(folder + "EJSD_THUG.npy", EJSD_THUG)
    save(folder + "G_EJSD_THUG.npy", G_EJSD_THUG)
    save(folder + "T_EJSD_THUG.npy", T_EJSD_THUG)
    save(folder + "A_HOP_THUG.npy", A_HOP_THUG)
    save(folder + "N_UNIQUE_THUG.npy", N_UNIQUE_THUG)

    save(folder + "THETA_ESS_THUG_AR.npy", THETA_ESS_THUG_AR)
    save(folder + "U_ESS_THUG_AR.npy", U_ESS_THUG_AR)
    save(folder + "ESS_THUG_AR.npy", ESS_THUG_AR)
    save(folder + "A_THUG_AR.npy", A_THUG_AR)
    save(folder + "RMSE_THUG_AR.npy", RMSE_THUG_AR)
    save(folder + "EJSD_THUG_AR.npy", EJSD_THUG_AR)
    save(folder + "G_EJSD_THUG_AR.npy", G_EJSD_THUG_AR)
    save(folder + "T_EJSD_THUG_AR.npy", T_EJSD_THUG_AR)
    save(folder + "A_HOP_THUG_AR.npy", A_HOP_THUG_AR)
    save(folder + "N_UNIQUE_THUG_AR.npy", N_UNIQUE_THUG_AR)