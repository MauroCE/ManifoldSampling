import time
import numpy as np
from numpy import zeros, diag, eye, log, vstack, sqrt, save, hstack
from numpy.linalg import solve, norm
from numpy.random import uniform, rand
from scipy.stats import multivariate_normal
import tensorflow_probability as tfp


def multiESS(X, b='sqroot', Noffsets=10, Nb=None):
    """
    Compute multivariate effective sample size of a single Markov chain X,
    using the multivariate dependence structure of the process.
    X: MCMC samples of shape (n, p)
    n: number of samples
    p: number of parameters
    b: specifies the batch size for estimation of the covariance matrix in
       Markov chain CLT. It can take a numeric value between 1 and n/2, or a
       char value between:
    'sqroot'    b=floor(n^(1/2)) (for chains with slow mixing time; default)
    'cuberoot'  b=floor(n^(1/3)) (for chains with fast mixing time)
    'lESS'      pick the b that produces the lowest effective sample size
                for a number of b ranging from n^(1/4) to n/max(20,p); this
                is a conservative choice
    If n is not divisible by b Sigma is recomputed for up to Noffsets subsets
    of the data with different offsets, and the output mESS is the average over
    the effective sample sizes obtained for different offsets.
    Nb specifies the number of values of b to test when b='less'
    (default NB=200). This option is unused for other choices of b.
    Original source: https://github.com/lacerbi/multiESS
    Reference:
    Vats, D., Flegal, J. M., & Jones, G. L. "Multivariate Output Analysis
    for Markov chain Monte Carlo", arXiv preprint arXiv:1512.07713 (2015).
    """

    # MCMC samples and parameters
    n, p = X.shape

    if p > n:
        raise ValueError(
            "More dimensions than data points, cannot compute effective "
            "sample size.")

    # Input check for batch size B
    if isinstance(b, str):
        if b not in ['sqroot', 'cuberoot', 'less']:
            raise ValueError(
                "Unknown string for batch size. Allowed arguments are "
                "'sqroot', 'cuberoot' and 'lESS'.")
        if b != 'less' and Nb is not None:
            raise Warning(
                "Nonempty parameter NB will be ignored (NB is used "
                "only with 'lESS' batch size B).")
    else:
        if not 1. < b < (n / 2):
            raise ValueError(
                "The batch size B needs to be between 1 and N/2.")

    # Compute multiESS for the chain
    mESS = multiESS_chain(X, n, p, b, Noffsets, Nb)

    return mESS


def multiESS_chain(Xi, n, p, b, Noffsets, Nb):
    """
    Compute multiESS for a MCMC chain.
    """

    if b == 'sqroot':
        b = [int(np.floor(n ** (1. / 2)))]
    elif b == 'cuberoot':
        b = [int(np.floor(n ** (1. / 3)))]
    elif b == 'less':
        b_min = np.floor(n ** (1. / 4))
        b_max = max(np.floor(n / max(p, 20)), np.floor(np.sqrt(n)))
        if Nb is None:
            Nb = 200
        # Try NB log-spaced values of B from B_MIN to B_MAX
        b = set(map(int, np.round(np.exp(
            np.linspace(np.log(b_min), np.log(b_max), Nb)))))

    # Sample mean
    theta = np.mean(Xi, axis=0)
    # Determinant of sample covariance matrix
    if p == 1:
        detLambda = np.cov(Xi.T)
    else:
        detLambda = np.linalg.det(np.cov(Xi.T))

    # Compute mESS
    mESS_i = []
    for bi in b:
        mESS_i.append(multiESS_batch(Xi, n, p, theta, detLambda, bi, Noffsets))
    # Return lowest mESS
    mESS = np.min(mESS_i)

    return mESS


def multiESS_batch(Xi, n, p, theta, detLambda, b, Noffsets):
    """
    Compute multiESS for a given batch size B.
    """

    # Compute batch estimator for SIGMA
    a = int(np.floor(n / b))
    Sigma = np.zeros((p, p))
    offsets = np.sort(list(set(map(int, np.round(
        np.linspace(0, n - np.dot(a, b), Noffsets))))))

    for j in offsets:
        # Swapped a, b in reshape compared to the original code.
        Y = Xi[j + np.arange(a * b), :].reshape((a, b, p))
        Ybar = np.squeeze(np.mean(Y, axis=1))
        Z = Ybar - theta
        for i in range(a):
            if p == 1:
                Sigma += Z[i] ** 2
            else:
                Sigma += Z[i][np.newaxis, :].T * Z[i]

    Sigma = (Sigma * b) / (a - 1) / len(offsets)
    mESS = n * (detLambda / np.linalg.det(Sigma)) ** (1. / p)

    return mESS


def HugTangential(x0, T, B, N, alpha, q, logpi, grad_log_pi):
    """
    Spherical Hug. Notice that it doesn't matter whether we use the gradient of pi or 
    grad log pi to tilt the velocity.
    """
    # Grab dimension, initialize storage for samples & acceptances
    samples, acceptances = x0, zeros(N)
    for i in range(N):
        v0s = q.rvs()                    # Draw velocity spherically
        g = grad_log_pi(x0)              # Compute gradient at x0
        g = g / norm(g)                  # Normalize
        v0 = v0s - alpha * g * (g @ v0s) # Tilt velocity
        v, x = v0, x0                    # Housekeeping
        logu = log(rand())               # Acceptance ratio
        delta = T / B                    # Compute step size

        for _ in range(B):
            x = x + delta*v/2           # Move to midpoint
            g = grad_log_pi(x)          # Compute gradient at midpoint
            ghat = g / norm(g)          # Normalize 
            v = v - 2*(v @ ghat) * ghat # Reflect velocity using midpoint gradient
            x = x + delta*v/2           # Move from midpoint to end-point
        # Unsqueeze the velocity
        g = grad_log_pi(x)
        g = g / norm(g)
        v = v + (alpha / (1 - alpha)) * g * (g @ v)
        # In the acceptance ratio must use spherical velocities!! Hence v0s and the unsqueezed v
        if logu <= logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0s):
            samples = vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
        else:
            samples = vstack((samples, x0))
            acceptances[i] = 0         # Rejected
    return samples[1:], acceptances


def Hug(x0, T, B, N, q, logpi, grad_log_pi):
    samples, acceptances = x0, zeros(N)
    for i in range(N):
        # Draw velocity
        v0 = q.rvs()
        # Housekeeping
        v, x = v0, x0
        # Acceptance ratio
        logu = log(rand())
        # Compute step size
        delta = T / B

        for _ in range(B):
            # Move
            x = x + delta*v/2 
            # Reflect
            g = grad_log_pi(x)
            ghat = g / norm(g)
            v = v - 2*(v @ ghat) * ghat
            # Move
            x = x + delta*v/2

        if logu <= logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0):
            samples = vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
        else:
            samples = vstack((samples, x0))
            acceptances[i] = 0         # Rejected
    return samples[1:], acceptances


def Hop(x, lam, k, logpi, grad_log_pi):
    d = len(x)
    # Compute matrix square root
    mu_sq = k * lam
    mu = sqrt(mu_sq)
    lam_sq = lam**2
    # Compute normalized gradient at x
    gx = grad_log_pi(x)
    ngx = norm(gx)
    ghatx = gx / ngx
    # Sample from standard MVN
    u = multivariate_normal(zeros(d), eye(d)).rvs()
    # Transform to new sample
    y = x + ((mu*u + (lam - mu) * ghatx * (ghatx @ u)) / sqrt(max(1.0, ngx**2)))
    # Compute stuff at y
    gy = grad_log_pi(y)
    ngy = norm(gy)
    # Acceptance probability
    logr = logpi(y) - logpi(x) 
    logr += d * (log(ngy) - log(ngx))
    logr -= (norm(y - x)**2) * (ngy**2 - ngx**2) / (2*mu_sq)
    logr -= 0.5 * (((y - x) @ gy)**2 - ((y - x) @ gx)**2) * ((1 / lam_sq) - (1 / mu_sq))
    # Accept or reject
    if log(rand()) <= min(0, logr):
        # Accept
        return y, 1.0
    else:
        # Reject - stay where you are
        return x, 0.0

def run_hug_hop(N):
    """Runs M iterations (cycles) of Hug and Hop. N is the number of Hug sub-iterations
    per iteration. In Hug&Hop paper N = 1. """
    samples = x = x0
    accept1 = np.array([0])
    while samples.shape[0] <= M:
        hug_samples, a1 = Hug(x, T, B, N, q, logpi, grad_log_pi)    # N iterations of Hug
        x, _  = Hop(hug_samples[-1], lam, k, logpi, grad_log_pi)   # 1 iteration of Hop
        samples = vstack((samples, hug_samples, x))             # Store N + 1 samples
        accept1 = hstack((accept1, a1))
    return samples[1:M+1], np.mean(accept1[1:])


def run_thug_hop(N, alpha):
    """Runs M iterations (cycles) of Hug and Hop. N is the number of Hug sub-iterations
    per iteration. In Hug&Hop paper N = 1. """
    samples = x = x0
    accept1 = np.array([0])
    while samples.shape[0] <= M:
        thug_samples, a1 = HugTangential(x, T, B, N, alpha, q, logpi, grad_log_pi)    # N iterations of Hug
        x, _  = Hop(thug_samples[-1], lam, kappa, logpi, grad_log_pi)   # 1 iteration of Hop
        samples = vstack((samples, thug_samples, x))             # Store N + 1 samples
        accept1 = hstack((accept1, a1))
    return samples[1:M+1], np.mean(accept1[1:])


# Target Distribution
d = 2
mu = zeros(d)
scale = 0.005
Sigma0 = diag([1.0, 5.0])
Sigma = scale * Sigma0
target = multivariate_normal(mu, Sigma)
logpi = target.logpdf
grad_log_pi = lambda xy: - solve(Sigma, xy - mu)
q = multivariate_normal(zeros(d), eye(d))
x0 = target.rvs()

# Settings
T = 5.0              # Total integration time for HUG/THUG
B = 5                # Number of bounces per HUG/THUG step
M = 10000            # Number of kernel iterations
n_runs = 10
lam = 2.0            # \lambda parameter for HOP
kappa = 0.25             # \kappa parameter for HOP
alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
Ns = [1, 10] #, 30, 50, 70, 100]

ESS_THUG = zeros((n_runs, len(Ns), len(alphas)))
ESS_logpi_THUG = zeros((n_runs, len(Ns), len(alphas)))
ESS_comp1_THUG = zeros((n_runs, len(Ns), len(alphas)))
ESS_comp2_THUG = zeros((n_runs, len(Ns), len(alphas)))

ESS_HUG  = zeros((n_runs, len(Ns)))
ESS_logpi_HUG = zeros((n_runs, len(Ns)))
ESS_comp1_HUG = zeros((n_runs, len(Ns)))
ESS_comp2_HUG = zeros((n_runs, len(Ns)))

A_THUG = zeros((len(Ns), len(alphas)))
A_HUG  = zeros(len(Ns))
for j, N in enumerate(Ns):
    for i in range(n_runs):
        # THUG for various alphas
        for k, alpha in enumerate(alphas):
            t_samples, ta = run_thug_hop(N, alpha)
            ESS_THUG[i, j, k] = multiESS(t_samples)
            ESS_logpi_THUG[i, j, k] = tfp.mcmc.effective_sample_size(target.logpdf(t_samples)).numpy()
            ess_comps_thug = tfp.mcmc.effective_sample_size(t_samples).numpy()
            ESS_comp1_THUG[i, j, k] = ess_comps_thug[0]
            ESS_comp2_THUG[i, j, k] = ess_comps_thug[1]
            A_THUG[j, k] += ta / n_runs
        # HUG
        h_samples, ha = run_hug_hop(N)
        ESS_HUG[i, j] = multiESS(h_samples)
        ESS_logpi_HUG[i, j] = tfp.mcmc.effective_sample_size(target.logpdf(h_samples)).numpy()
        ess_comps_hug = tfp.mcmc.effective_sample_size(h_samples).numpy()
        ESS_comp1_HUG[i, j] = ess_comps_hug[0]
        ESS_comp2_HUG[i, j] = ess_comps_hug[1]
        A_HUG[j] += ha / n_runs


save("ESS_THUG.npy", ESS_THUG)
save("ESS_LOGPI_THUG.npy", ESS_logpi_THUG)
save("ESS_COMP1_THUG.npy", ESS_comp1_THUG)
save("ESS_COMP2_THUG.npy", ESS_comp2_THUG)
save("ATHUG.npy", A_THUG)

save("ESS_HUG.npy", ESS_HUG)
save("ESS_LOGPI_HUG.npy", ESS_logpi_HUG)
save("ESS_COMP1_HUG.npy", ESS_comp1_HUG)
save("ESS_COMP2_HUG.npy", ESS_comp2_HUG)
save("AHUG.npy", A_HUG)