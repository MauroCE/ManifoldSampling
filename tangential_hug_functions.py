import numpy as np
from numpy.linalg import norm, solve, cholesky, det
from scipy.linalg import solve_triangular
from numpy.random import rand
from scipy.stats import multivariate_normal
# numpy version 1.19.5 worked


def HugTangential(x0, T, B, n, alpha, q, logpi, grad_log_pi):
    """
    Spherical Hug. Notice that it doesn't matter whether we use the gradient of pi or 
    grad log pi to tilt the velocity.
    """
    # Grab dimension, initialize storage for samples & acceptances
    samples, acceptances = x0, np.zeros(n)
    for i in range(n):
        v0s = q.rvs()                    # Draw velocity spherically
        g = grad_log_pi(x0)              # Compute gradient at x0
        g = g / norm(g)                  # Normalize
        v0 = v0s - alpha * g * (g @ v0s) # Tilt velocity
        v, x = v0, x0                    # Housekeeping
        logu = np.log(rand())            # Acceptance ratio
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
            samples = np.vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
        else:
            samples = np.vstack((samples, x0))
            acceptances[i] = 0         # Rejected
    return samples[1:], acceptances


def Hug(x0, T, B, n, q, logpi, grad_log_pi):
    """
    Standard Hug Kernel.
    """
    samples, acceptances = x0, np.zeros(n)
    for i in range(n):
        # Draw velocity
        v0 = q.rvs()
        # Housekeeping
        v, x = v0, x0
        # Acceptance ratio
        logu = np.log(rand())
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
            samples = np.vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
        else:
            samples = np.vstack((samples, x0))
            acceptances[i] = 0         # Rejected
    return samples[1:], acceptances


def HugPC(x0, T, B, S, n, q, logpi, grad_log_pi):
    """
    Preconditioned Hug Kernel. S is a function that takes a position x and returns
    sample covariance matrix of dimension d. 
    """
    samples, acceptances = x0, np.zeros(n)
    for i in range(n):
        # Draw velocity
        v0 = q.rvs()
        # Housekeeping
        v, x = v0, x0
        # Acceptance ratio
        logu = np.log(rand())
        # Compute step size
        delta = T / B

        for _ in range(B):
            # Move
            x = x + delta*v/2 
            # Reflect
            g = grad_log_pi(x)
            Sg = S(x) @ g
            v = v - 2*(v @ g) * Sg / (g @ Sg)
            #v = v - 2*(v @ g) * Sg / (g @ Sg)
            # Move
            x = x + delta*v/2

        if logu <= logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0):
            samples = np.vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
        else:
            samples = np.vstack((samples, x0))
            acceptances[i] = 0         # Rejected
    return samples[1:], acceptances


def NoAR(x00, T, B, n, alphas, q_sample, grad_log_pi):
    """
    Hug and THug without Accept-Reject.
    """
    # Grab dimension
    d = len(x00)
    v_sphericals = np.vstack([q_sample() for _ in range(n)])
    t_finals = np.zeros((len(alphas), d))
    ### HUG
    x0 = x00
    for i in range(n):
        # Draw velocity
        v0 = v_sphericals[i]
        
        # Housekeeping
        v = v0
        x = x0
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
        x0 = x
    h_final = x
    ### THUG
    for ix, alpha in enumerate(alphas):
        x0 = x00
        for i in range(n):
            # Draw velocity spherically
            v0s = v_sphericals[i]
            # Compute normalized gradient at x0
            g = grad_log_pi(x0)
            g = g / norm(g)
            # Tilt velocity
            v0 = v0s - alpha * g * (g @ v0s)
            # Housekeeping
            v = v0
            x = x0
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
            x0 = x
        t_finals[ix] = x
    return h_final, t_finals   # Grab only final point for Hug, grab final points for each α for THUG


def HugAcceleration(x0, T, B, n, q_sample, logq, logpi, grad_log_pi, Sigma):
    """
    Hug but doesn't bounce, simply approx the acceleration.
    """
    samples = x0
    acceptances = np.zeros(n)
    for i in range(n):
        # Draw velocity
        v0 = q_sample()
        
        # Housekeeping
        v = v0
        x = x0
        # Acceptance ratio
        logu = np.log(rand())
        # Compute step size
        delta = T / B

        for _ in range(B):
            # Move
            logpi1 = logpi(x)
            g = grad_log_pi(x)
            x = x + delta*v/2 
            # Approx Hessian
            g_new = grad_log_pi(x)
            g_new_hat = g_new / norm(g_new)
            #v = v -8 * (logpi(x) - logpi1) * g / ((norm(g)**2) * delta)
            v = v - delta * v @ solve(Sigma, v) * g_new_hat / norm(g_new)
            # Move
            x = x + delta*v/2

        if logu <= logpi(x) + logq(v) - logpi(x0) - logq(v0):
            samples = np.vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
        else:
            samples = np.vstack((samples, x0))
            acceptances[i] = 0         # Rejected
    return samples, acceptances


def Hop(x, lam, k, logpi, grad_log_pi):
    """
    One iteration of the Hop kernel.
    """
    d = len(x)
    # Compute matrix square root
    mu_sq = k * lam
    mu = np.sqrt(mu_sq)
    lam_sq = lam**2
    # Compute normalized gradient at x
    gx = grad_log_pi(x)
    ngx = norm(gx)
    ghatx = gx / ngx
    # Sample from standard MVN
    u = multivariate_normal(np.zeros(d), np.eye(d)).rvs()
    # Transform to new sample
    y = x + ((mu*u + (lam - mu) * ghatx * (ghatx @ u)) / np.sqrt(max(1.0, ngx**2)))
    # Compute stuff at y
    gy = grad_log_pi(y)
    ngy = norm(gy)
    # Acceptance probability
    logr = logpi(y) - logpi(x) 
    logr += d * (np.log(ngy) - np.log(ngx))
    logr -= (norm(y - x)**2) * (ngy**2 - ngx**2) / (2*mu_sq)
    logr -= 0.5 * (((y - x) @ gy)**2 - ((y - x) @ gx)**2) * ((1 / lam_sq) - (1 / mu_sq))
    # Accept or reject
    if np.log(np.random.rand()) <= min(0, logr):
        # Accept
        return y, 1.0
    else:
        # Reject - stay where you are
        return x, 0.0


def HopPC(x, S, lam, k, logpi, grad_log_pi):
    """
    Preconditioned Hop Kernel. One single iteration. 
    Needs a function S that returns a local coviariance matrix. Will actually end up using
    its cholesky decomposition. See https://math.stackexchange.com/q/4204891/318854.
    """
    d = len(x)
    # Compute parameters
    mu_sq = k * lam
    mu = np.sqrt(mu_sq)
    lam_sq = lam**2
    # Compute matrix square root of S. Since this is one iteration, can cache it.
    # recall `L = cholesky(S)` returns a lower-triangular matrix L such that
    # L @ L.T is equal to S. However, to make calculations easier, we will use
    # L.T.
    # MUST LOOK INTO USING cho_factor and cho_solve.
    Ax = cholesky(S(x)).T
    # Compute normalized gradient at x
    gx = grad_log_pi(x)
    # Sample from standard MVN
    u = multivariate_normal(np.zeros(d), np.eye(d)).rvs()
    # Compute new gradient variable f = Ag
    Agx = Ax @ gx
    nAgx = norm(Agx)
    Au = Ax @ u
    y = x + ((mu * Au + (lam - mu) * Agx * (Agx @ Au)) / np.sqrt(max(1.0, nAgx**2)))
    # Compute stuff at y
    gy = grad_log_pi(y)
    # Compute its preconditioned norm
    Ay = cholesky(S(y)).T
    Agy = Ay @ gy
    nAgy = norm(Agy)
    # Compute hessians times gradients
    # Notice we can solve using Cholesky decomposition. 
    ymx = y - x
    Hy_ymx = Hess(Ay, ymx)
    Hx_ymx = Hess(Ax, ymx)
    # Acceptance probability
    logr = logpi(y) - logpi(x) 
    logr += d * (np.log(nAgy) - np.log(nAgx))
    logr += 0.5 * (np.log(det(Ax)) - np.log(det(Ay)))
    logr -= ymx @ ((nAgx**2) * Hx_ymx - (nAgy**2) * Hy_ymx) / (2 * mu_sq)
    logr -= 0.5 * ((1 / lam_sq) - (1 / mu_sq)) * ((ymx @ gy)**2 - (ymx @ gx)**2)
    # Accept or reject
    if np.log(np.random.rand()) <= min(0, logr):
        # Accept
        return y, 1.0
    else:
        # Reject - stay where you are
        return x, 0.0


def Hess(A, x):
    """Given upper triangular matrix A which is transpose of Cholesky decomp of coviariance matrix S, and given
     a vector x, it computes H @ x where H is the hessian matrix corresponding to S (see HH paper). Usually H = -S^{-1}
    so we do `solve(-S, x)`. However this can fail, in which case.."""
    return -solve_triangular(A.T, solve_triangular(A, x, lower=False), lower=True)


def GradientHug(x0, T, B, q, logpi, grad_log_pi):
    """
    gradient wrong Hug Kernel.
    """
    v0 = q.rvs()
    x, v = x0, v0
    logu = np.log(rand())
    delta = T/B
    for _ in range(B):
        # Move
        x = x + delta* v/2 
        # Reflect using negated velocity
        g = grad_log_pi(x)
        ghat = g / norm(g)
        v = - v + 2*(v @ ghat) * ghat
        # Move
        x = x + delta*v/2

    if logu <= logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0):
        return x, 1    # Accepted
    else:
        return x0, 0 # Rejected


def GradientHugPC(x0, T, B, S, q, logpi, grad_log_pi):
    """
    Preconditioned Gradient Hug. Notice we might need to do a different preconditioning 
    than for Hug. For now, we are using the same.
    """
    v0 = q.rvs()
    x, v = x0, v0
    logu = np.log(rand())
    delta = T/B
    for _ in range(B):
        # Move
        x = x + delta* v/2 
        # Reflect using negated velocity
        g = grad_log_pi(x)
        Sg = S(x) @ g
        v = - v + 2*(v @ g) * Sg / (g @ Sg)
        # Move
        x = x + delta*v/2

    if logu <= logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0):
        return x, 1    # Accepted
    else:
        return x0, 0 # Rejected


def run_hug_hop(x0, T, B, N, lam, k, q, logpi, grad_log_pi):
    """HUG + HOP"""
    # Housekeeping
    samples = x = x0
    ahug, ahop = np.zeros(N), np.zeros(N)
    for _ in range(N):
        x_hug, acc_hug = Hug(x, T, B, 1, q, logpi, grad_log_pi)
        x, acc_hop = Hop(x_hug.flatten(), lam, k, logpi, grad_log_pi)
        # Housekeeping
        samples, ahug[_], ahop[_] = np.vstack((samples, x_hug, x)), acc_hug, acc_hop
    return samples[1:], ahug, ahop


def run_hug_hop_pc(x0, T, B, S, N, lam, k, q, logpi, grad_log_pi):
    """HUG + HOP PRECONDITIONED"""
    samples = x = x0
    ahug, ahop = np.zeros(N), np.zeros(N)
    for _ in range(N):
        x_hug, acc_hug = HugPC(x, T, B, S, 1, q, logpi, grad_log_pi)
        x, acc_hop = HopPC(x_hug.flatten(), S, lam, k, logpi, grad_log_pi)
        samples, ahug[_], ahop[_] = np.vstack((samples, x_hug, x)), acc_hug, acc_hop
    return samples[1:], ahug, ahop


def run_hug_gradient(x0, T, B, N, q, logpi, grad_log_pi):
    """HUG + GRADIENT HUG"""
    # Housekeeping
    samples = x = x0
    ahug, aghug = np.zeros(N), np.zeros(N)
    for _ in range(N):
        x_hug, acc_hug = Hug(x, T, B, 1, q, logpi, grad_log_pi)
        x, acc_ghug = GradientHug(x_hug.flatten(), T, B, q, logpi, grad_log_pi)
        # Housekeeping
        samples, ahug[_], aghug[_] = np.vstack((samples, x_hug, x)), acc_hug, acc_ghug
    return samples[1:], ahug, aghug


def run_thug_hop(x0, T, B, N, alpha, lam, k, q, logpi, grad_log_pi):
    """THUG + HOP"""
    # Housekeeping
    samples = x = x0
    athug, ahop = np.zeros(N), np.zeros(N)
    for _ in range(N):
        x_thug, acc_thug = HugTangential(x, T, B, 1, alpha, q, logpi, grad_log_pi)
        x, acc_hop = Hop(x_thug.flatten(), lam, k, logpi, grad_log_pi)
        # Housekeeping
        samples, athug[_], ahop[_] = np.vstack((samples, x_thug, x)), acc_thug, acc_hop
    return samples[1:], athug, ahop


def run_thug_gradient(x0, T, B, N, alpha, q, logpi, grad_log_pi):
    """THUG + GRADIENT HUG"""
    # Housekeeping
    samples = x = x0
    athug, aghug = np.zeros(N), np.zeros(N)
    for _ in range(N):
        x_thug, acc_thug = HugTangential(x, T, B, 1, alpha, q, logpi, grad_log_pi)
        x, acc_ghug = GradientHug(x_thug.flatten(), T, B, q, logpi, grad_log_pi)
        samples, athug[_], aghug[_] = np.vstack((samples, x_thug, x)), acc_thug, acc_ghug
    return samples[1:], athug, aghug


def run_hug_gradient_pc(x0, T, B, S, N, q, logpi, grad_log_pi):
    """HUG + GRADIENT HUG PC"""
    samples = x = x0
    ahugpc, aghugpc = np.zeros(N), np.zeros(N)
    for _ in range(N):
        x_hugpc, acc_hugpc = HugPC(x, T, B, S, 1, q, logpi, grad_log_pi)
        x, acc_ghugpc = GradientHugPC(x_hugpc.flatten(), T, B, S, q, logpi, grad_log_pi)
        samples, ahugpc[_], aghugpc[_] = np.vstack((samples, x_hugpc, x)), acc_hugpc, acc_ghugpc
    return samples[1:], ahugpc, aghugpc
