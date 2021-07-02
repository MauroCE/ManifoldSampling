import numpy as np
from numpy.linalg import norm
from numpy.random import rand
from scipy.stats import multivariate_normal


def HugTangential(x0, T, B, n, alpha, q_sample, logq, logpi, grad_log_pi):
    """
    Spherical Hug. Notice that it doesn't matter whether we use the gradient of pi or 
    grad log pi to tilt the velocity.
    """
    # grab dimension
    d = len(x0)
    samples = x0
    acceptances = np.zeros(n)
    for i in range(n):
        # Draw velocity spherically
        v0s = q_sample()
        # Compute normalized gradient at x0
        g = grad_log_pi(x0)
        g = g / norm(g)
        # Tilt velocity
        v0 = v0s - alpha * g * (g @ v0s)
        # Housekeeping
        v = v0
        x = x0
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
        # Unsqueeze the velocity
        g = grad_log_pi(x)
        g = g / norm(g)
        v = v + (alpha / (1 - alpha)) * g * (g @ v)
        # Need to compute the density of the tilted proposals
        standard_MVN = multivariate_normal(np.zeros(d), np.eye(d))
        logq0 = lambda xy: standard_MVN.logpdf(xy) 
        logq  = lambda xy: standard_MVN.logpdf(xy)
        if logu <= logpi(x) + logq(v) - logpi(x0) - logq0(v0):
            samples = np.vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
        else:
            samples = np.vstack((samples, x0))
            acceptances[i] = 0         # Rejected
    return samples, acceptances


def Hug(x0, T, B, n, q_sample, logq, logpi, grad_log_pi):
    """
    Standard Hug Kernel. This uses no preconditioning. This is ONE STEP of the hug kernel.
    Returns a triplet (x, v, a) where x is the new sample, v is the velocity at the new sample
    and a is a binary flag indicating successful acceptance (a=1) or rejection (a=0).

    x0 : Numpy Array
         Point from which to do 1 step fo Hug. Basically the difference between self.x0 and
         x0 is that self.x0 is the starting point of the whole algorithm, while x0 is just the 
         starting point for this Hug.
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
            x = x + delta*v/2 
            # Reflect
            g = grad_log_pi(x)
            ghat = g / norm(g)
            v = v - 2*(v @ ghat) * ghat
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