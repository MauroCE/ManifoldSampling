"""
Experiment 73: p-Dimensional Ellipsoid example where we compute the Expected Squared Jump Distance
(ESJD) as the acceptance probability times the squared distance for each proposal. Notice that we 
use the acceptance probability and not the acceptance ratio, i.e. the value is clipped between 0 and
1. We use p=10 and this experiment is fundamentally the same as Experiment 71, except that we try
to parallelize the grid-evaluation. Indeed we compute the mean ESJD over a grid of alpha and delta
values to figure out if using larger alphas, for a "challenging" epsilon, it is possible to obtain
better mean ESJD than using alpha=0. 
"""
from itertools import product
import numpy as np
from numpy import save
from scipy.stats import multivariate_normal as MVN
from scipy.optimize import root, fsolve, minimize
from numpy.linalg import norm, solve, inv, det
from numpy import log, zeros, eye, exp, cos, sin, pi, diag
from scipy.stats import uniform as udist
from numpy.random import rand
from matplotlib.colors import ListedColormap
from warnings import catch_warnings, filterwarnings
from scipy.linalg import qr
from copy import deepcopy
from scipy.linalg import solve_triangular, qr, lstsq
from multiprocessing import Pool
from Manifolds.GeneralizedEllipse import GeneralizedEllipse

# Define constraint function
p = 10
μ = zeros(p)
diagonal = np.array([0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) #np.array([100, 100, *np.ones(p-2)])#np.concatenate((np.full(p//2, 100), np.ones(p - (p//2))))
Σ = diag(diagonal)
π = MVN(μ, Σ)
f = π.logpdf

# Ellipsoid
z0 = -23
ellipse = GeneralizedEllipse(μ, Σ, exp(z0))

def HugTangentialMultivariate(x0, T, B, N, α, q, logpi, jac, method='qr'):
    """Multidimensional Tangential Hug sampler. Two possible methods:
    - 'qr': projects onto row space of Jacobian using QR decomposition.
    - 'linear': solves a linear system to project.
    """
    EJSD_AP = [np.nan for i in range(N)]
    assert method == 'qr' or method == 'linear' or method == 'lstsq'
    def qr_project(v, J):
        """Projects using QR decomposition."""
        Q, _ = qr(J.T, mode='economic')
        return Q.dot((Q.T.dot(v)))
    def linear_project(v, J):
        """Projects by solving linear system."""
        return J.T.dot(solve(J.dot(J.T), J.dot(v)))
    def lstsq_project(v, J):
        """Projects using scipy's Least Squares Routine."""
        return J.T.dot(lstsq(J.T, v)[0])
    if method == 'qr':
        project = qr_project
    elif method == 'linear':
        project = linear_project
    else:
        project = lstsq_project
    # Jacobian function raising an error for RuntimeWarning
    def safe_jac(x):
        """Raises an error when a RuntimeWarning appears."""
        while catch_warnings():
            filterwarnings('error')
            try:
                return jac(x)
            except RuntimeWarning:
                raise ValueError("Jacobian computation failed due to Runtime Warning.")
    samples, acceptances = x0, np.zeros(N)
    # Compute initial Jacobian. 
    for i in range(N):
        v0s = q.rvs()
        # Squeeze
        v0 = v0s - α * project(v0s, safe_jac(x0)) #jac(x0))
        v, x = v0, x0
        logu = np.log(rand())
        δ = T / B
        for _ in range(B):
            xmid = x + δ*v/2
            v = v - 2 * project(v, safe_jac(xmid)) #jac(x))
            x = xmid + δ*v/2
        # Unsqueeze
        v = v + (α / (1 - α)) * project(v, safe_jac(x)) #jac(x))
        # In the acceptance ratio must use spherical velocities!! Hence v0s and the unsqueezed v
        logar = logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0s)
        ar = exp(logar)
        EJSD_AP[i] = np.clip(ar, a_min=0.0, a_max=1.0) * (norm(x - x0)**2)
        if logu <= logar:
            x0 = x
    return EJSD_AP


q = MVN(zeros(p), eye(p))
normalize = lambda a: a / np.nanmax(a)

def generate_starting_points(ϵs, n_tries=10):
    x0s = np.zeros((len(ϵs), p))
    for i in range(len(ϵs)):
        j = 0
        while j <= n_tries:
            j += 1
            with catch_warnings():
                filterwarnings('error')
                try: 
                    x0 = ellipse.sample(advanced=True)
                    x0s[i, :] = x0
                    break
                except RuntimeWarning:
                    print(f"Can't find {i}th point. Trying again.")
                    continue
            if j == n_tries:
                raise ValueError("Couldn't find point.")
    return x0s

def generate_logpi(ϵ):
    def logkernel(xi):
        return -norm(f(xi) - z0)**2/(2*(ϵ**2)) - np.log(ϵ)
    # Logpi
    logpi = lambda xi: logkernel(xi)
    return logpi

x0s = generate_starting_points(np.zeros(1))

B = 10
N_GRID = 40
N = 10000
ϵ = 0.0001  ### smaller now!
αs_eps0001 = np.linspace(start=0.0, stop=1.0, num=N_GRID, endpoint=False)
δs_eps0001 = np.geomspace(start=0.05, stop=2.0, num=N_GRID, endpoint=True)
n_cores = 8

args = product(x0s, [B], [N], [ϵ], αs_eps0001, δs_eps0001)

def run_exp_in_parallel(arguments):
    x0, B, N, ϵ, α, δ = arguments
    logpi = generate_logpi(ϵ)
    jac = lambda xi: ellipse.Q(xi).T
    mean_ESJD = np.mean(HugTangentialMultivariate(x0, δ*B, B, N, α, q, logpi, jac))
    return mean_ESJD

if __name__ == "__main__":
    try:
        with Pool(n_cores) as p:
            results = p.map(run_exp_in_parallel, args)
    except KeyboardInterrupt:
        p.terminate()
    except Exception as e:
        print('Exception occurred: ', e)
        p.terminate()
    finally:
        p.join()

    folder = 'experiment73/'    

    save(folder + 'results.npy', np.array(results).reshape(N_GRID, N_GRID))
    save(folder + 'alphas.npy', αs_eps0001)
    save(folder + 'deltas.npy', δs_eps0001)
