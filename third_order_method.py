import numpy as np
from numpy import log, zeros, vstack
from numpy.random import rand
from numpy.linalg import norm, solve
from scipy.stats import multivariate_normal as MVN

### INTEGRATORS

def true_ruth_integrator(x0, v0, T, B, gradf, hessf):
    """True Ruth Integrator for a general contour.
    x0 : Starting position.
    v0 : Startign velocity.
    T  : Integration time.
    B  : Number of bounces. (delta = T / B)
    gradf : Function computing gradient of the deterministic function at an input x.
    hessf : Function computing hessian of the deterministic function at an input x.
    """
    x, v = x0, v0
    delta = T / B
    for _ in range(B):
        g = gradf(x); ng = norm(g); ghat = g / ng
        v1 = v - (7/24) * delta * ((v @ (hessf(x) @ v)) / ng) * ghat
        x1 = x + (2/3) * delta * v1

        g1 = gradf(x1); ng1 = norm(g1); ghat1 = g1 / ng1
        v2 = v1 - (3/4) * delta * ((v1 @ (hessf(x1) @ v1)) / ng1) * ghat1
        x2 = x1 - (2/3) * delta * v2

        g2 = gradf(x2); ng2 = norm(g2); ghat2 = g2 / ng2
        v  = v2 + (1/24) * delta * ((v2 @ (hessf(x2) @ v2)) / ng2) * ghat2
        x  = x2 + delta * v
    return x, v


def fourth_order_explicit(x0, v0, T, B, gradf, hessf):
    """Sans Serna book p.109"""
    x, v = x0, v0
    δ = T / B
    for _ in range(B):
        v1 = v
        x1 = x + δ*v1; g1 = gradf(x1); g1n = norm(g1); g1hat = g1 / g1n;
        v2 = v1 + (1/24)*δ*((v1 @ (hessf(x1) @ v1)) / g1n) * g1hat
        x2 = x1 - (2/3)*δ*v2; g2 = gradf(x2); g2n = norm(g2); g2hat = g2 / g2n;
        v3 = v2 - (3/4)*δ*((v2 @ (hessf(x2) @ v2)) / g2n) * g2hat
        x3 = x2 + (2/3)*δ*v3; g3 = gradf(x3); g3n = norm(g3); g3hat = g3 / g3n;
        v4 = v3 - (7/12)*δ*((v3 @ (hessf(x3) @ v3)) / g3n) * g3hat
        x4 = x3 + (2/3)*δ*v4; g4 = gradf(x4); g4n = norm(g4); g4hat = g4 / g4n;
        v5 = v4 - (3/4)*δ*((v4 @ (hessf(x4) @ v4)) / g4n) * g4hat
        x5 = x4 - (2/3)*δ*v5; g5 = gradf(x5); g5n = norm(g5); g5hat = g5 / g5n;
        v  = v5 + (1/24)*δ*((v5 @ (hessf(x5) @ v5)) / g5n) * g5hat
        x  = x5 + δ * v
    return x, v


def true_hug_integrator(x0, v0, δ, grad_f, hess_f):
    """True Hug Integrator."""
    x, v = x0, v0
    x = x + δ*v/2
    g = grad_f(x); gn = norm(g); ghat = g / gn;
    v = v - δ * ((v @ (hess_f(x) @ v)) / gn) * ghat
    x = x + δ*v/2
    return x, v

def approx_hug_integrator(x0, v0, δ, grad_f):
    x, v = x0, v0
    x = x + δ*v/2
    g = grad_f(x); ghat = g / norm(g);
    v = v - 2 * (v @ ghat) * ghat
    x = x + δ*v/2
    return x, v

def concatenate_true_hug_integrators(x0, v0, δ, grad_f, hess_f):
    """Concatenate 3 true hug integrators using technique by Reich."""
    c1 = 1 / (2 - 2**(1/3))
    c2 = - 2**(1/3) / (2 - 2**(1/3))
    x1, v1 = true_hug_integrator(x0, v0, c1 * δ, grad_f, hess_f)
    x2, v2 = true_hug_integrator(x1, v1, c2 * δ, grad_f, hess_f)
    x3, v3 = true_hug_integrator(x2, v2, c1 * δ, grad_f, hess_f)
    return x3, v3


def concatenate_approx_hug_integrators(x0, v0, δ, grad_f):
    """Concatenate 3 approx hug integrators using technique by Reich."""
    c1 = 1 / (2 - 2**(1/3))
    c2 = - 2**(1/3) / (2 - 2**(1/3))
    x1, v1 = approx_hug_integrator(x0, v0, c1 * δ, grad_f)
    x2, v2 = approx_hug_integrator(x1, v1, c2 * δ, grad_f)
    x3, v3 = approx_hug_integrator(x2, v2, c1 * δ, grad_f)
    return x3, v3



def true_ruth_integrator_gaussian(x0, v0, T, B, Sigma, mu):
    """True Ruth Integrator for Gaussian Target."""
    x, v = x0, v0
    delta = T / B
    gradf = lambda x: - solve(Sigma, x - mu)
    for _ in range(B):
        g = gradf(x); ng = norm(g); ghat = g / ng
        v1 = v + (7/24) * delta * ((v @ solve(Sigma, v)) / ng) * ghat
        x1 = x + (2/3) * delta * v1

        g1 = gradf(x1); ng1 = norm(g1); ghat1 = g1 / ng1
        v2 = v1 + (3/4) * delta * ((v1 @ solve(Sigma, v1)) / ng1) * ghat1
        x2 = x1 - (2/3) * delta * v2

        g2 = gradf(x2); ng2 = norm(g2); ghat2 = g2 / ng2
        v  = v2 - (1/24) * delta * ((v2 @ solve(Sigma, v2)) / ng2) * ghat2
        x  = x2 + delta * v
    return x, v


def approx_ruth_integrator(x0, v0, T, B, gradf):
    """Approximate Ruth Integrator. This uses the final step."""
    x, v = x0, v0
    delta = T / B
    for _ in range(B):
        g = gradf(x); ng = norm(g); ghat = g / ng
        x1 = x + (7/24) * delta * v
        g1 = gradf(x1); ng1 = norm(g1); ghat1 = g1 / ng1
        v1 = v - ghat * (ghat1 @ v)

        x2 = x1 + (2/3) * delta * v1
        g2 = gradf(x2); ng2 = norm(g2); ghat2 = g2 / ng2
        v2 = v1 - (9/8) * ghat2 * (ghat2 @ v1)

        x3 = x2 - (1/24) * delta * v2
        g3 = gradf(x3); ng3 = norm(g3); ghat3 = g3 / ng3
        v  = v2 - ghat3 * (ghat3 @ v2)
        x  = x3 + delta * v
    return x, v

def approx_ruth_integrator_earlystop(x0, v0, T, B, gradf):
    """Approximate Ruth Integrator stopping at x_t''."""
    x, v = x0, v0
    delta = T / B
    g = gradf(x); ng = norm(g); ghat = g / ng
    for _ in range(B):
        x1 = x + (7/24) * delta * v
        g1 = gradf(x1); ng1 = norm(g1); ghat1 = g1 / ng1
        v1 = v - ghat * (ghat1 @ v)

        x2 = x1 + (2/3) * delta * v1
        g2 = gradf(x2); ng2 = norm(g2); ghat2 = g2 / ng2
        v2 = v1 - (9/8) * ghat2 * (ghat2 @ v1)

        x = x2 - (1/24) * delta * v2
        g = gradf(x); ng = norm(g); ghat = g / ng
        v  = v2 - ghat * (ghat @ v2)
    return x, v

### SAMPLERS

def true_ruth_sampler(x0, T, B, N, q, logpi, gradf, hessf):
    """Equivalent of Hug sampler. This is a reversible version."""
    samples, acceptances = x0, zeros(N)
    for i in range(N):
        v0s = q.rvs()
        ####### TODO: REMOVE THIS
        g0 = gradf(x0)
        g0 = g0 / norm(g0)
        v0 = v0s - (v0s @ g0) * g0
        v0 = (v0 / norm(v0)) * norm(v0s)
        logu = log(rand())
        x, v = true_ruth_integrator(x0, v0, T, B, gradf, hessf)
        if logu <= logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0):
            samples = vstack((samples, x))
            acceptances[i] = 1
            x0 = x
        else:
            samples = vstack((samples, x0))
            acceptances[i] = 0
        return samples[1:], acceptances


def true_ruth_sampler_thug(x0, T, B, N, alpha, q, logpi, gradf, hessf):
    """Equivalent of THug sampler. This is a reversible version."""
    samples, acceptances = x0, zeros(N)
    for i in range(N):
        v0s = q.rvs()
        # Squeeze
        g0 = gradf(x0)
        g0 = g0 / norm(g0)
        v0 = v0s - alpha * (v0s @ g0) * g0
        logu = log(rand())
        x, v = true_ruth_integrator(x0, v0, T, B, gradf, hessf)
        # Unsqueeze
        g = gradf(x)
        g = g / norm(g)
        vs = v + (alpha / (1 - alpha)) * (v @ g) * g
        if logu <= logpi(x) + q.logpdf(vs) - logpi(x0) - q.logpdf(v0s):
            samples = vstack((samples, x))
            acceptances[i] = 1
            x0 = x
        else:
            samples = vstack((samples, x0))
            acceptances[i] = 0
        return samples[1:], acceptances




### UTILS

mse = lambda x, y: np.mean((x - y)**2)

def check_involution(x0, v0, func):
    """Checks if a certain integrator is an involution or not."""
    def involution(x, v):
        xx, vv = func(x, v)
        return xx, -vv
    x0new, v0new = involution(*involution(x0, v0))
    return mse(x0new, x0), mse(v0new, v0)


def new_integrator(x, v, δ, gradf):
    x = x + (1/8)*δ*v; g = gradf(x); ghat = g / norm(g);
    v = v - 2*(v @ ghat)*ghat
    x = x + (3/8)*δ*v; g = gradf(x); ghat = g / norm(g);
    v = v - 2*(v @ ghat)*ghat
    x = x + (3/8)*δ*v; g = gradf(x); ghat = g / norm(g);
    v = v - 2*(v @ ghat)*ghat
    x = x + (1/8)*δ*v
    return x, v


def compute_coefficients(b1, b3):
    b2 = 1 - b1 - b3
    a1 = b1 * b3 / (b1 + b3)
    a4 = a1
    a2 = 0.5 - a1
    a3 = a2
    b1_num = b1 / a1
    b2_num = b2 / (a1 + a2 - b1)
    b3_num = b3 / (a1 + a2 + a3 - b1 - b2)
    return a1, a2, a3, a4, b1, b2, b3, b1_num, b2_num, b3_num

def new_general_integrator(x, v, δ, gradf, b1=0.25, b3=0.25):
    a1, a2, a3, a4, b1, b2, b3, b1_num, b2_num, b3_num = compute_coefficients(b1, b3)
    # Integrator
    x = x + a1*δ*v; g = gradf(x); ghat = g / norm(g);
    v = v - b1_num*(v @ ghat)*ghat
    x = x + a2*δ*v; g = gradf(x); ghat = g / norm(g);
    v = v - b2_num*(v @ ghat)*ghat
    x = x + a3*δ*v; g = gradf(x); ghat = g / norm(g);
    v = v - b3_num*(v @ ghat)*ghat
    x = x + a4*δ*v
    return x, v

# def true_ruth_integrator(x0, v0, grad_f=lambda x: x, delta=1.0):
#     x, v = x0, v0
#     for _ in range(B):
#         g = grad_f(x); ng = norm(g); ghat = g / ng
#         v1 = v + (7/24) * delta * ((v @ solve(Σ, v)) / ng) * ghat
#         x1 = x + (2/3) * delta * v1

#         g1 = grad_f(x1); ng1 = norm(g1); ghat1 = g1 / ng1
#         v2 = v1 + (3/4) * delta * ((v1 @ solve(Σ, v1)) / ng1) * ghat1
#         x2 = x1 - (2/3) * delta * v2

#         g2 = grad_f(x2); ng2 = norm(g2); ghat2 = g2 / ng2
#         v  = v2 - (1/24) * delta * ((v2 @ solve(Σ, v2)) / ng2) * ghat2
#         x  = x2 + delta * v
#     return x, v

# def ruth_integrator(x0, v0, delta, gnorm, B):
#     """Third order sympletic integrator by Ruth."""
#     x, v = x0, v0
#     g = gnorm(x)
#     for _ in range(B):
#         x1 = x + (7/24) * delta * v
#         v1 = v - g * (gnorm(x1) @ v)
#         x2 = x1 + (2/3) * delta * v1
#         v2 = v1 - (9/8) * gnorm(x2) * (gnorm(x2) @ v1)
#         x3 = x2 - (1/24) * delta * v2
#         v  = v2 - gnorm(x3) * (gnorm(x3) @ v2)
#         x  = x3 + delta * v
#     return x, v
