from numpy import array, exp, apply_along_axis, r_, vstack
from numpy import ones, zeros, log, diag, errstate, hstack
from numpy import isfinite
from numpy.linalg import norm
from numpy.random import default_rng, uniform, randn

from scipy.stats import norm as ndist
from scipy.optimize import fsolve

from warnings import catch_warnings, filterwarnings


def f(xi): 
    """Simulator. This is a deterministic function."""
    a, b, g, k, *z = xi
    z = array(z)
    return a + b*(1 + 0.8 * (1 - exp(-g * z)) / (1 + exp(-g * z))) * ((1 + z**2)**k) * z

def f_broadcast(xi_matrix):
    """Broadcasted version of f."""
    return apply_along_axis(f, 1, xi_matrix)

def fnorm(xi, ystar):
    """This function is h(xi) = |f(xi) - y*|. Basically the function defining Chang's manifold."""
    return norm(f(xi) - ystar)

def fnorm_broadcast(xi_matrix, ystar):
    """Broadcasted version of fnorm."""
    return norm(f_broadcast(xi_matrix) - ystar, axis=1)

def data_generator(theta, N, seed):
    """Generates data with a given random seed."""
    rng = default_rng(seed)
    z = rng.normal(size=N)
    return f(r_[theta, z])

def Jf_transpose(xi):
    """Jacobian function of f."""
    _, b, g, k, *z = xi
    z = array(z)
    return vstack((
        ones(len(z)),
        (1 + 0.8 * (1 - exp(-g * z)) / (1 + exp(-g * z))) * ((1 + z**2)**k) * z,
        8 * b * (z**2) * ((1 + z**2)**k) * exp(g*z) / (5 * (1 + exp(g*z))**2),
        b*z*((1+z**2)**k)*(1 + 9*exp(g*z))*log(1 + z**2) / (5*(1 + exp(g*z))),
        diag(b*((1+z**2)**(k-1))*(((18*k + 9)*(z**2) + 9)*exp(2*g*z) + (8*g*z**3 + (20*k + 10)*z**2 + 8*g*z + 10)*exp(g*z) + (2*k + 1)*z**2 + 1) / (5*(1 + exp(g*z))**2))
    ))

def grad_fnorm(xi, ystar):
    """Gradient of h(xi)."""
    return Jf_transpose(xi) @ (f(xi) - ystar)

def logprior(xi):
    theta, z = xi[:4], xi[4:]
    with errstate(divide='ignore'):
        return log(((0 <= theta) & (theta <= 10)).all().astype('float64')) + ndist.logpdf(z).sum()

def logprior_broadcast(xi_matrix):
    with errstate(divide='ignore'):
        return ((0 <= xi_matrix[:, :4]) & (xi_matrix[:, :4] <= 10)).all(axis=1).astype('float64') + ndist.logpdf(xi_matrix[:, 4:]).sum(axis=1)

def sample_prior(n_params, n_latents):
    """Sample from prior distribution over params and latents."""
    return r_[uniform(low=0.0, high=10.0, size=n_params), randn(n_latents)]

def log_epanechnikov_kernel(xi, epsilon, fnorm, ystar):
    u = fnorm(xi, ystar)
    with errstate(divide='ignore'):
        return log((3*(1 - (u**2 / (epsilon**2))) / (4*epsilon)) * float(u <= epsilon))

def new_point(n_params, n_latents, fnorm, y_star, max_iter, epsilon, threshold):
    """Generates a new point for a given epsilon. Notice this is a different function from experiment 34."""
    func = lambda xi: r_[fnorm(xi, y_star), zeros(n_params + n_latents-1)]
    log_abc_posterior = lambda xi, epsilon: logprior(xi) + log_epanechnikov_kernel(xi, epsilon, fnorm, y_star)
    i = 0
    with catch_warnings():
        filterwarnings('error')
        while i <= max_iter:
            i += 1
            try: 
                theta_guess = uniform(0.0, 10.0, size=4)
                guess = hstack((theta_guess, zeros(n_latents)))
                point = fsolve(func, guess)
                if not isfinite([log_abc_posterior(point, epsilon)]):
                    pass
                else:
                    if fnorm(point, y_star) < threshold:
                        return point
                    else:
                        continue
            except RuntimeWarning:
                continue
