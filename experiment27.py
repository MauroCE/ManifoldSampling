# Experiment 27: Zappa on 2D Gaussian Toy example
import numpy as np
from numpy import diag, zeros, exp, log, eye, vstack, save
from scipy.stats import multivariate_normal as MVN
from numpy.linalg import norm, solve
from numpy.random import rand
from time import time

from Manifolds.GeneralizedEllipse import GeneralizedEllipse
from Zappa.zappa import zappa_sampling_multivariate, zappa_step_EJSD_deterministic
from utils import ESS, ESS_univariate

from multiprocessing import Pool
import warnings
warnings.filterwarnings("error")
from itertools import product


def experiment(x00, sigma):
    logp = lambda xi: MVN(zeros(d-1), (sigma**2) * eye(d-1)).logpdf(xi)
    ### COMMON VARIABLES
    v = MVN(zeros(d-1), eye(d-1)).rvs(N)
    log_uniforms = log(rand(N))     # Log uniforms for zappa
    ### STORAGE (HUG + HOP)
    samples = x00
    ap = 0.0               # Acceptance probability for zappa
    ejsd = 0.0             # EJSD
    tot_n_grad = 0.0
    ### ZAPPA ON EXACT MANIFOLD
    x = x00
    for i in range(N):
        v_i = np.array([v[i]])
        x, a, e, n_grad = zappa_step_EJSD_deterministic(x, v_i, log_uniforms[i], manifold, logf, logp, sigma, tol, a_guess)
        tot_n_grad += n_grad
        samples = vstack((samples, x))
        ap += a * 100 / N
        ejsd += e / N
    # COMPUTE ESS AND OTHER METRICS FOR HUG
    samples = samples[1:]
    ess_univariate  = ESS_univariate(samples)
    ess_joint       = ESS(samples)
    ### RETURN EVERYTHING
    out = {
        'A': ap,
        'E': ejsd,
        'ESS': ess_univariate,
        'ESS_J': ess_joint,
        'N_GRAD': tot_n_grad
    }
    return out

        

# MVN defining the manifold
Sigma = diag([1.0, 5.0]) 
d = Sigma.shape[0]
mu = zeros(d)
target = MVN(mu, Sigma)

# Manifold
z0 = -2.9513586307684885
manifold = GeneralizedEllipse(mu, Sigma, exp(z0))

# Settings
N = 50000  
n_runs = 10
n_cores = 8
tol = 1.48e-08
a_guess = 1.0

# Target on the manifold
logf = lambda xi: - log(norm(solve(Sigma, xi)))


Ts = [10, 1, 0.1, 0.01] 
n_T = len(Ts)


initial_time = time()
initial_points = vstack([manifold.sample() for _ in range(n_runs)])
run_indices = np.arange(n_runs).tolist()  # One for each run
results = []

args = product(initial_points, [Ts])

# FUNCTION TO RUN IN PARALLEL
def my_function(arguments):
    x00, Ts = arguments
    out_dict = {}
    for T in Ts:
            scale = T / (5 * 2)  # Delta / 2
            out_dict['{}'.format(T)] = experiment(x00, scale)
    return out_dict


if __name__ == "__main__":
    # Store results
    ESS_ZAPPA       = zeros((n_runs, n_T, d))     # Univariate ESS for each dim
    ESS_JOINT_ZAPPA = zeros((n_runs, n_T))        # multiESS on whole chain
    A_ZAPPA         = zeros((n_runs, n_T))        # Acceptance probability
    EJSD_ZAPPA      = zeros((n_runs, n_T))        # Full EJSD 
    N_GRAD_ZAPPA    = zeros((n_runs, n_T))        # Total number of gradients

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
        for j, T in enumerate(Ts):
            # Store results
            try:
                ESS_ZAPPA[i, j, :]    = results[i]['{}'.format(T)]['ESS']
            except TypeError as e:
                print("uESS: ", results[i]['{}'.format(T)]['ESS'])
                ESS_ZAPPA[i, j, :] = 0.0
            try:
                ESS_JOINT_ZAPPA[i, j] = results[i]['{}'.format(T)]['ESS_J']
            except TypeError as e:
                print("jESS: ", results[i]['{}'.format(T)]['ESS_J'])
                ESS_JOINT_ZAPPA[i, j] = 0.0
            A_ZAPPA[i, j]         = results[i]['{}'.format(T)]['A']
            EJSD_ZAPPA[i, j]      = results[i]['{}'.format(T)]['E']
            N_GRAD_ZAPPA[i, j]    = results[i]['{}'.format(T)]['N_GRAD']

    print("Total time: ", time() - initial_time)

    # Save results
    folder = "experiment27/"

    save(folder + "TS.npy", Ts)
    save(folder + "TIME.npy", np.array([time() - initial_time]))
    save(folder + "D.npy", d)

    save(folder + "ESS_ZAPPA.npy", ESS_ZAPPA)
    save(folder + "ESS_JOINT_ZAPPA.npy", ESS_JOINT_ZAPPA)
    save(folder + "A_ZAPPA.npy", A_ZAPPA)
    save(folder + "EJSD_ZAPPA.npy", EJSD_ZAPPA)
    save(folder + "N_GRAD.npy", N_GRAD_ZAPPA)

