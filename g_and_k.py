import autograd.numpy as np
from autograd.numpy import exp
from autograd.numpy.linalg import norm 
from autograd import grad

from numpy.random import randn, rand
from numpy import log, zeros, eye, vstack
from scipy.stats import norm as ndist
from scipy.stats import multivariate_normal as MVN
from scipy.optimize import fsolve
from statsmodels.tsa.stattools import acf 

from utils import ESS_univariate, ESS, n_unique
from tangential_hug_functions import HugTangentialStepEJSD_Deterministic
from tangential_hug_functions import Hop_Deterministic
from tangential_hug_functions import HugStepEJSD_Deterministic


transform = lambda a, b, g, k, z: a + b*(1 + 0.8 * (1 - exp(-g * z)) / (1 + exp(-g * z))) * ((1 + z**2)**k) * z

def f(thetau):
    """Deterministic function for distance manifold. f:R^5 -> R """
    a_param, b_param, k_param, *z = thetau  # Latents are standard normal variables
    z = np.array(z)
    out = transform(a_param, b_param, g_true, k_param, z)
    return norm(out - y_star)


def experiment(x00, T1, T2, N, alphas, nlags):
    """Runs Hug+Hop and THUG+HOP using the same velocities and the same random seeds."""
    ### COMMON VARIABLES
    v = q.rvs(N)
    log_uniforms1 = log(rand(N))     # Log uniforms for the HUG kernels
    log_uniforms2 = log(rand(N))     # Log uniforms for the HOP kernel
    u = MVN(zeros(5), eye(5)).rvs(N) # Original velocities for HOP kernel
    ### STORAGE (HUG + HOP)
    hh = x00              # Initial sample
    ahh1 = 0.0            # Acceptance probability for HUG kernel
    ahh2 = 0.0            # Acceptance probability for HOP kernel (when used with HUG)
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
        y, a1, e, eg, et = HugStepEJSD_Deterministic(x, v[i], log_uniforms1[i], T1, B, q, log_abc_posterior, grad_function)
        x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, log_abc_posterior, grad_function)
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
            y, a1, e, eg, et = HugTangentialStepEJSD_Deterministic(x, v[i], log_uniforms1[i], T2, B, alpha, q, log_abc_posterior, grad_function)
            x, a2 = Hop_Deterministic(y, u[i], log_uniforms2[i], lam, kappa, log_abc_posterior, grad_function)
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
            'T': T1,
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
            'T': T2,
            'SAMPLES': th
        }
    }
    return out


# Prior, Kernel and Posterior
def logprior(thetau):
    """Log prior distribution."""
    with np.errstate(divide='ignore'):
        return log((abs(thetau[:3]) <= 10).all().astype('float32')) + ndist.logpdf(thetau[-2:]).sum()
    
def log_uniform_kernel(xi, epsilon):
    """Log density of uniform kernel. """
    with np.errstate(divide='ignore'):
        return log(f(xi) <= epsilon)
    
def log_abc_posterior(xi):
    """Log density of ABC posterior. Product of (param-latent) prior and uniform kernel."""
    return logprior(xi) + log_uniform_kernel(xi, epsilon)


if __name__ == "__main__":
    ### Settings
    m = 2                    # Number of latents/observations
    alphas = [0.95]
    n_alphas = len(alphas)
    epsilon = 0.1
    lam = epsilon / 4
    kappa = 0.25
    T1 = T2 = 0.01
    B = 5
    N = 100
    nlags = 20

    # True Parameter Value
    a_true, b_true, g_true, k_true = 3.0, 1.0, 2.0, 0.5
    theta0 = np.array([a_true, b_true, k_true])          

    # Generate Initial Data
    y_star = transform(*theta0, *randn(m))

    # Gradient function
    grad_function = grad(f)

    # Proposal for HUG/THUG
    q = MVN(zeros(5), eye(5))

    # Initial Guess
    guess = np.array([1.0, 1.0, 1.0, 0.0, 0.0]) 
    func = lambda xi: np.r_[f(xi), 0.0, 0.0, 0.0, 0.0]  # Append 0, 0 to make fsolve work.
    xi0 = fsolve(func, guess)

    # Run experiment
    out = experiment(xi0, T1, T2, N, alphas, 20)
