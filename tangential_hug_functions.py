from Manifolds.GeneralizedEllipse import GeneralizedEllipse
import numpy as np
from numpy.lib.twodim_base import eye
from numpy.linalg import norm, solve, cholesky, det
from scipy.linalg import solve_triangular
from numpy.random import rand
from scipy.stats import multivariate_normal
from Manifolds.RotatedEllipse import RotatedEllipse
from Zappa.zappa import zappa_step_accept, zappa_step_acceptPC
import time
from numpy.random import uniform
# numpy version 1.19.5 worked


def HugTangential(x0, T, B, N, alpha, q, logpi, grad_log_pi):
    """
    Tangential Hug. Notice that it doesn't matter whether we use the gradient of pi or 
    grad log pi to tilt the velocity.
    """
    # Grab dimension, initialize storage for samples & acceptances
    samples, acceptances = x0, np.zeros(N)
    for i in range(N):
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



def HugTangential_EJSD(x0, T, B, N, alpha, q, logpi, grad_log_pi):
    """
    Same as HugTangential but also computes EJSD.
    """
    # Grab dimension, initialize storage for samples & acceptances
    samples, acceptances = x0, np.zeros(N)
    ejsd = np.zeros(N)
    for i in range(N):
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
        loga = logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0s)
        a = min(1.0, np.exp(loga))
        ejsd[i] = a * norm(x0 - x)**2
        if logu <= loga:
            samples = np.vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
        else:
            samples = np.vstack((samples, x0))
            acceptances[i] = 0         # Rejected
    return samples[1:], acceptances, ejsd



def HugTangentialAR(x0, T, B, N, alpha, prob, q, logpi, grad_log_pi):
    """
    Non-reversible Tangential Hug. At every iteration we sample a variable
    w ~ U(0, 1) and with probability prob = 0.75 we use an AR process with \rho=1
    (i.e. we keep exactly the previous velocity). Otherwise we use an AR process 
    with \rho=0 (i.e. we completely refresh the velocity). The AR process is
            V_{t+1} = \rho V_t + \sqrt{1 - \rho^2} W_t
    where V_t is the previous velocity, W_t ~ N(0, 1) and |\rho| <= 1.
    > At the moment, I am doing the AR process on the spherical velocity.
    """
    # Grab dimension, initialize storage for samples & acceptances
    samples, acceptances = x0, np.zeros(N)
    # Before starting, sample a spherical velocity
    v0s = q.rvs()
    for i in range(N):
        # With probability prob keep the same velocity, otherwise refresh completely
        if uniform() > prob: 
            v0s = q.rvs()
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
        vs = v + (alpha / (1 - alpha)) * g * (g @ v)
        # In the acceptance ratio must use spherical velocities!! Hence v0s and the unsqueezed v
        if logu <= logpi(x) + q.logpdf(vs) - logpi(x0) - q.logpdf(v0s):
            samples = np.vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
            # IMPORTANT: Notice that we do not save the tilted velocity because
            # we need to tilt it with the NEW GRADIENT!!
        else:
            samples = np.vstack((samples, x0))
            acceptances[i] = 0         # Rejected
            # Upon rejection, I need to negate the velocity
            v0s = -v0s
    return samples[1:], acceptances



def HugTangentialAR_EJSD(x0, T, B, N, alpha, prob, q, logpi, grad_log_pi):
    """
    Same as HugTangentialAR but this also computes the EJSD.
    """
    # Grab dimension, initialize storage for samples & acceptances
    samples, acceptances = x0, np.zeros(N)
    ejsd = np.zeros(N)
    # Before starting, sample a spherical velocity
    v0s = q.rvs()
    for i in range(N):
        # With probability prob keep the same velocity, otherwise refresh completely
        if uniform() > prob: 
            v0s = q.rvs()
        g0 = grad_log_pi(x0)              # Compute gradient at x0
        g0 = g0 / norm(g0)                  # Normalize
        v0 = v0s - alpha * g0 * (g0 @ v0s) # Tilt velocity
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
        vs = v + (alpha / (1 - alpha)) * g * (g @ v)
        # Acceptance probability
        loga = logpi(x) + q.logpdf(vs) - logpi(x0) - q.logpdf(v0s)
        a = min(1.0, np.exp(loga))
        ejsd[i] = a * norm(x0 - x)**2 
        # In the acceptance ratio must use spherical velocities!! Hence v0s and the unsqueezed v
        if logu <= loga:
            samples = np.vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
            # IMPORTANT: Notice that we do not save the tilted velocity because
            # we need to tilt it with the NEW GRADIENT!!
        else:
            samples = np.vstack((samples, x0))
            acceptances[i] = 0         # Rejected
            # Upon rejection, I need to negate the velocity
            v0s = -v0s
    return samples[1:], acceptances, ejsd



def HugTangentialARrho(x0, T, B, N, alpha, rho, q, logpi, grad_log_pi):
    """
    Non-reversible Tangential Hug. Here we use the full AR process.
    """
    # Grab dimension, initialize storage for samples & acceptances
    samples, acceptances = x0, np.zeros(N)
    assert abs(rho) <=1, "You must provide rho with |rho| <= 1."
    # Before starting, sample a spherical velocity
    v0s = q.rvs()
    for i in range(N):
        # With probability prob keep the same velocity, otherwise refresh completely
        v0s = rho*v0s + np.sqrt(1 - rho**2)*q.rvs()
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
        vs = v + (alpha / (1 - alpha)) * g * (g @ v)
        # In the acceptance ratio must use spherical velocities!! Hence v0s and the unsqueezed v
        if logu <= logpi(x) + q.logpdf(vs) - logpi(x0) - q.logpdf(v0s):
            samples = np.vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
            # IMPORTANT: Notice that we do not save the tilted velocity because
            # we need to tilt it with the NEW GRADIENT!!
        else:
            samples = np.vstack((samples, x0))
            acceptances[i] = 0         # Rejected
            # Upon rejection, I need to negate the velocity
            v0s = -v0s
    return samples[1:], acceptances




def HugTangentialARrho_EJSD(x0, T, B, N, alpha, rho, q, logpi, grad_log_pi):
    """
    Same as HugTangentialARrho but computes EJSD.
    """
    # Grab dimension, initialize storage for samples & acceptances
    samples, acceptances = x0, np.zeros(N)
    assert abs(rho) <=1, "You must provide rho with |rho| <= 1."
    ejsd = 0.0
    # Before starting, sample a spherical velocity
    v0s = q.rvs()
    for i in range(N):
        # With probability prob keep the same velocity, otherwise refresh completely
        v0s = rho*v0s + np.sqrt(1 - rho**2)*q.rvs()
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
        vs = v + (alpha / (1 - alpha)) * g * (g @ v)
        # In the acceptance ratio must use spherical velocities!! Hence v0s and the unsqueezed v
        loga = logpi(x) + q.logpdf(vs) - logpi(x0) - q.logpdf(v0s)
        a = min(1.0, np.exp(loga))
        ejsd += (a * norm(x0 - x)**2) / N
        if logu <= loga:
            samples = np.vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
            # IMPORTANT: Notice that we do not save the tilted velocity because
            # we need to tilt it with the NEW GRADIENT!!
        else:
            samples = np.vstack((samples, x0))
            acceptances[i] = 0         # Rejected
            # Upon rejection, I need to negate the velocity
            v0s = -v0s
    return samples[1:], acceptances, ejsd


def compare_HUG_THUG_THUGAR(x00, T, B, N, alpha, prob, q, logpi, grad_log_pi):
    """Compares the three algorithms using same noise."""
    velocities = q.rvs(N)
    logus = np.log(rand(N))
    #### HUG
    x0 = x00
    hug, ahug = x0, np.zeros(N)
    for i in range(N):
        # Draw velocity
        v0 = velocities[i]
        # Housekeeping
        v, x = v0, x0
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

        if logus[i] <= logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0):
            hug = np.vstack((hug, x))
            ahug[i] = 1         # Accepted!
            x0 = x
        else:
            hug = np.vstack((hug, x0))
            ahug[i] = 0         # Rejected
    #### THUG
    x0 = x00
    thug, athug = x0, np.zeros(N)
    for i in range(N):
        v0s = velocities[i]              # Draw velocity spherically
        g = grad_log_pi(x0)              # Compute gradient at x0
        g = g / norm(g)                  # Normalize
        v0 = v0s - alpha * g * (g @ v0s) # Tilt velocity
        v, x = v0, x0                    # Housekeeping
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
        if logus[i] <= logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0s):
            thug = np.vstack((thug, x))
            athug[i] = 1         # Accepted!
            x0 = x
        else:
            thug = np.vstack((thug, x0))
            athug[i] = 0         # Rejected
    #### THUG-AR
    x0 = x00
    thug_ar, athug_ar = x0, np.zeros(N)
    for i in range(N):
        # With probability prob keep the same velocity, otherwise refresh completely
        if uniform() > prob or i==0: 
            v0s = velocities[i]
        g = grad_log_pi(x0)              # Compute gradient at x0
        g = g / norm(g)                  # Normalize
        v0 = v0s - alpha * g * (g @ v0s) # Tilt velocity
        v, x = v0, x0                    # Housekeeping
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
        vs = v + (alpha / (1 - alpha)) * g * (g @ v)
        # In the acceptance ratio must use spherical velocities!! Hence v0s and the unsqueezed v
        if logus[i] <= logpi(x) + q.logpdf(vs) - logpi(x0) - q.logpdf(v0s):
            thug_ar = np.vstack((thug_ar, x))
            athug_ar[i] = 1         # Accepted!
            x0 = x
            # IMPORTANT: Notice that we do not save the tilted velocity because
            # we need to tilt it with the NEW GRADIENT!!
        else:
            thug_ar = np.vstack((thug_ar, x0))
            athug_ar[i] = 0         # Rejected
            # Upon rejection, I need to negate the velocity
            v0s = -v0s
    return hug, thug, thug_ar, ahug, athug, athug_ar


def compare_HUG_THUG_THUGAR_rho(x00, T, B, N, alpha, rho, q, logpi, grad_log_pi):
    """Compares the three algorithms using same noise."""
    velocities = q.rvs(N)
    logus = np.log(rand(N))
    #### HUG
    x0 = x00
    hug, ahug = x0, np.zeros(N)
    for i in range(N):
        # Draw velocity
        v0 = velocities[i]
        # Housekeeping
        v, x = v0, x0
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

        if logus[i] <= logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0):
            hug = np.vstack((hug, x))
            ahug[i] = 1         # Accepted!
            x0 = x
        else:
            hug = np.vstack((hug, x0))
            ahug[i] = 0         # Rejected
    #### THUG
    x0 = x00
    thug, athug = x0, np.zeros(N)
    for i in range(N):
        v0s = velocities[i]              # Draw velocity spherically
        g = grad_log_pi(x0)              # Compute gradient at x0
        g = g / norm(g)                  # Normalize
        v0 = v0s - alpha * g * (g @ v0s) # Tilt velocity
        v, x = v0, x0                    # Housekeeping
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
        if logus[i] <= logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0s):
            thug = np.vstack((thug, x))
            athug[i] = 1         # Accepted!
            x0 = x
        else:
            thug = np.vstack((thug, x0))
            athug[i] = 0         # Rejected
    #### THUG-AR
    x0 = x00
    thug_ar, athug_ar = x0, np.zeros(N)
    for i in range(N):
        # With probability prob keep the same velocity, otherwise refresh completely
        if i == 0:
            v0s = velocities[i]
        else:
            v0s = rho*v0s + np.sqrt(1 - rho**2)*velocities[i]
        g = grad_log_pi(x0)              # Compute gradient at x0
        g = g / norm(g)                  # Normalize
        v0 = v0s - alpha * g * (g @ v0s) # Tilt velocity
        v, x = v0, x0                    # Housekeeping
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
        vs = v + (alpha / (1 - alpha)) * g * (g @ v)
        # In the acceptance ratio must use spherical velocities!! Hence v0s and the unsqueezed v
        if logus[i] <= logpi(x) + q.logpdf(vs) - logpi(x0) - q.logpdf(v0s):
            thug_ar = np.vstack((thug_ar, x))
            athug_ar[i] = 1         # Accepted!
            x0 = x
            # IMPORTANT: Notice that we do not save the tilted velocity because
            # we need to tilt it with the NEW GRADIENT!!
        else:
            thug_ar = np.vstack((thug_ar, x0))
            athug_ar[i] = 0         # Rejected
            # Upon rejection, I need to negate the velocity
            v0s = -v0s
    return hug, thug, thug_ar, ahug, athug, athug_ar



def HugTangentialPC(x0, T, B, S, N, alpha, q, logpi, grad_log_pi):
    """
    Spherical Hug. Notice that it doesn't matter whether we use the gradient of pi or 
    grad log pi to tilt the velocity.
    """
    # Grab dimension, initialize storage for samples & acceptances
    samples, acceptances = x0, np.zeros(N)
    for i in range(N):
        v0s = q.rvs()                    # Draw velocity spherically
        g = grad_log_pi(x0)              # Compute gradient at x0
        Sg = S(x0) @ g
        v0 = v0s - alpha * (g @ v0s) * Sg / (g @ Sg) # Tilt velocity
        v, x = v0, x0                    # Housekeeping
        logu = np.log(rand())            # Acceptance ratio
        delta = T / B                    # Compute step size

        for _ in range(B):
            x = x + delta*v/2           # Move to midpoint
            g = grad_log_pi(x)          # Compute gradient at midpoint
            Sg = S(x) @ g
            v = v - 2*(v @ g) * Sg / (g @ Sg) # Reflect velocity using midpoint gradient
            x = x + delta*v/2           # Move from midpoint to end-point
        # Unsqueeze the velocity
        g = grad_log_pi(x)
        Sg = S(x) @ g
        v = v + (alpha / (1 - alpha)) * (v @ g) * Sg / (g @ Sg)
        # In the acceptance ratio must use spherical velocities!! Hence v0s and the unsqueezed v
        if logu <= logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0s):
            samples = np.vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
        else:
            samples = np.vstack((samples, x0))
            acceptances[i] = 0         # Rejected
    return samples[1:], acceptances


def Hug(x0, T, B, N, q, logpi, grad_log_pi):
    """
    Standard Hug Kernel.
    """
    samples, acceptances = x0, np.zeros(N)
    for i in range(N):
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


def Hug_EJSD(x0, T, B, N, q, logpi, grad_log_pi):
    """
    Hug kernel but also outputs EJSD.
    """
    samples, acceptances = x0, np.zeros(N)
    ejsd = np.zeros(N)
    for i in range(N):
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
        loga = logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0)
        a = min(1.0, np.exp(loga))
        ejsd += (a * norm(x0 - x)**2) / N
        if logu <= loga:
            samples = np.vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
        else:
            samples = np.vstack((samples, x0))
            acceptances[i] = 0         # Rejected
    return samples[1:], acceptances, ejsd


def HugAR(x0, T, B, N, prob, q, logpi, grad_log_pi):
    """
    Hug with degenerate AR process.
    """
    samples, acceptances = x0, np.zeros(N)
    v0 = q.rvs()     # Draw initial spherical velocity
    for i in range(N):
        if uniform() > prob:
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
            # Since this is a non-reversible algorithm, we must negate the velocity at the end
            v0 = -v0
    return samples[1:], acceptances


def HugAR_EJSD(x0, T, B, N, prob, q, logpi, grad_log_pi):
    """
    Same as HugAR but also outputs EJSD. Notice it outputs the averaged (i.e. Expected)
    JSD already. Again, this should be used on its own, not with Hop. 
    """
    samples, acceptances = x0, np.zeros(N)
    ejsd = np.zeros(N)
    v0 = q.rvs()     # Draw initial spherical velocity
    for i in range(N):
        if uniform() > prob:
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

        loga = logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0)
        a = min(1.0, np.exp(loga))
        ejsd += (a * norm(x0 - x)**2) / N
        if logu <= loga:
            samples = np.vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
        else:
            samples = np.vstack((samples, x0))
            acceptances[i] = 0         # Rejected
            # Since this is a non-reversible algorithm, we must negate the velocity at the end
            v0 = -v0
    return samples[1:], acceptances, ejsd



def HugStepEJSD(x0, T, B, q, logpi, grad_log_pi):
    """
    One step of HUG kernel computing Expected Squared Jump Distace.
    """
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
    loga = logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0)  # Log acceptance probability
    a = min(1.0, np.exp(loga))   # Acceptance probability
    # ESJD overall
    ESJD = a * norm(x0 - x)**2
    # ESJD for gradient and tangent component
    g0 = grad_log_pi(x0)
    g0hat = g0 / norm(g0)
    gx = grad_log_pi(x)
    gxhat = gx / norm(gx)
    x0grad = x0 @ g0hat
    xgrad = x @ gxhat
    ESJD_GRAD = a * (x0grad- xgrad)**2
    x0tan = norm(x0 - x0grad * g0hat)
    xtan = norm(x - xgrad * gxhat)
    ESJD_TAN = a * (x0tan - xtan)**2
    if logu <= loga:
        return x, 1, ESJD, ESJD_GRAD, ESJD_TAN
    else:
        return x0, 0, ESJD, ESJD_GRAD, ESJD_TAN


def HugTangentialStepEJSD_AR(x0, v0s, v0, T, B, alpha, q, logpi, grad_log_pi):
    """
    One step of THUG computing ESJD.
    """
    v0s = q.rvs()                    # Draw velocity spherically
    g0 = grad_log_pi(x0)              # Compute gradient at x0
    g0 = g0 / norm(g0)                  # Normalize
    v0 = v0s - alpha * g0 * (g0 @ v0s) # Tilt velocity
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
    loga = logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0s)
    a = min(1.0, np.exp(loga))
    # Compute EJSD
    ESJD = a * norm(x0 - x)**2
    x0grad = x0 @ g0
    xgrad  = x  @ g
    ESJD_GRAD = a * (x0grad- xgrad)**2
    x0tan = norm(x0 - x0grad * g0)
    xtan = norm(x - xgrad * g)
    ESJD_TAN = a * (x0tan - xtan)**2
    if logu <= loga:
        return x, 1, ESJD, ESJD_GRAD, ESJD_TAN
    else:
        return x0, 0, ESJD, ESJD_GRAD, ESJD_TAN



def HugTangentialStepEJSD(x0, T, B, alpha, q, logpi, grad_log_pi):
    """
    One step of THUG computing ESJD.
    """
    v0s = q.rvs()                    # Draw velocity spherically
    g0 = grad_log_pi(x0)              # Compute gradient at x0
    g0 = g0 / norm(g0)                  # Normalize
    v0 = v0s - alpha * g0 * (g0 @ v0s) # Tilt velocity
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
    loga = logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0s)
    a = min(1.0, np.exp(loga))
    # Compute EJSD
    ESJD = a * norm(x0 - x)**2
    x0grad = x0 @ g0
    xgrad  = x  @ g
    ESJD_GRAD = a * (x0grad- xgrad)**2
    x0tan = norm(x0 - x0grad * g0)
    xtan = norm(x - xgrad * g)
    ESJD_TAN = a * (x0tan - xtan)**2
    if logu <= loga:
        return x, 1, ESJD, ESJD_GRAD, ESJD_TAN
    else:
        return x0, 0, ESJD, ESJD_GRAD, ESJD_TAN


def HugPC(x0, T, B, S, N, q, logpi, grad_log_pi):
    """
    Preconditioned Hug Kernel. S is a function that takes a position x and returns
    sample covariance matrix of dimension d. 
    """
    samples, acceptances = x0, np.zeros(N)
    for i in range(N):
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


def NoAR(x00, T, B, N, alphas, q_sample, grad_log_pi):
    """
    Hug and THug without Accept-Reject.
    """
    # Grab dimension
    d = len(x00)
    v_sphericals = np.vstack([q_sample() for _ in range(N)])
    t_finals = np.zeros((len(alphas), d))
    ### HUG
    x0 = x00
    for i in range(N):
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
        for i in range(N):
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


def HugAcceleration(x0, T, B, N, q_sample, logq, logpi, grad_log_pi, Sigma):
    """
    Hug but doesn't bounce, simply approx the acceleration.
    """
    samples = x0
    acceptances = np.zeros(N)
    for i in range(N):
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
    Sx = S(x)
    ATx = cholesky(Sx)
    Ax = ATx.T
    # Compute normalized gradient at x
    gx = grad_log_pi(x)
    gSg = gx @ (Sx @ gx)

    # Sample from standard MVN
    u = multivariate_normal(np.zeros(d), np.eye(d)).rvs()
    # Compute new gradient variable f = Ag
    Agx = Ax @ gx
    nAgx = norm(Agx)
    ATu = ATx @ u
    y = x + ((mu * ATu + (lam - mu) * ATx @ Agx * (gx @ ATu) / gSg) / np.sqrt(max(1.0, nAgx**2)))
    #y = x + ((mu * Au + (lam - mu) * Agx * (Agx @ Au)) / np.sqrt(max(1.0, nAgx**2)))
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

def run_thug_hop_pc(x0, T, B, S, N, alpha, lam, k, q, logpi, grad_log_pi):
    """THUG + HOP PRECONDITIONED"""
    samples = x = x0
    athug, ahop = np.zeros(N), np.zeros(N)
    for _ in range(N):
        x_thug, acc_thug = HugTangentialPC(x, T, B, S, 1, alpha, q, logpi, grad_log_pi)
        x, acc_hop = HopPC(x_thug.flatten(), S, lam, k, logpi, grad_log_pi)
        samples, athug[_], ahop[_] = np.vstack((samples, x_thug, x)), acc_thug, acc_hop
    return samples[1:], athug, ahop


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


def run_thug_gradient_pc(x0, T, B, S, N, alpha, q, logpi, grad_log_pi):
    """THUG + GRADIENT HUG PRECONDITIONED"""
    samples = x = x0
    athug, aghug = np.zeros(N), np.zeros(N)
    for _ in range(N):
        x_thug, acc_thug = HugTangentialPC(x, T, B, S, 1, alpha, q, logpi, grad_log_pi)
        x, acc_ghug = GradientHugPC(x_thug.flatten(), T, B, S, q, logpi, grad_log_pi)
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


def run(kernel1, kernel2, x0, N, args1, args2):
    """Runs a cycle of kernels with kernel1 and kernel2."""
    # Sanity check
    assert args1['logpi'] == args2['logpi']
    assert args1['grad_log_pi'] == args2['grad_log_pi']
    t = time.time()
    # Storage
    samples = x = x0
    accept1 = np.zeros(N)
    accept2 = np.zeros(N)
    for _ in range(N):
        # Cycle kernels
        y, a1 = kernel1(x, **args1)
        x, a2 = kernel2(y.flatten(), **args2)
        # Storage
        samples, accept1[_], accept2[_] = np.vstack((samples, y, x)), a1, a2
    return samples[1:], accept1, accept2, time.time() - t


def cycle_zappa(kernel2, x0, N, m, target, p, tol, a_guess, args2):
    """Runs a cycle of Zappa and kernel 2."""
    t = time.time()
    # Storage
    samples = x = x0
    accept1 = np.zeros(N)
    accept2 = np.zeros(N)
    for _ in range(N):
        # Cycle kernels
        manifold = GeneralizedEllipse(target.mean, target.cov, target.pdf(x))
        y, a1 = zappa_step_accept(x, manifold, p, m, tol, a_guess)
        x, a2 = kernel2(y.flatten(), **args2)
        samples, accept1[_], accept2[_] = np.vstack((samples, y, x)), a1, a2
    return samples[1:], accept1, accept2, time.time() - t


def cycle_zappaPC(kernel2, x0, N, A, m, target, p, tol, a_guess, args2):
    """Runs a cycle of ZappaPC and kernel 2."""
    t = time.time()
    manifold = GeneralizedEllipse(target.mean, target.cov, target.pdf(x0))
    d = manifold.get_dimension()  # Dimension of tangent space
    # Storage
    samples = x = x0
    accept1 = np.zeros(N)
    accept2 = np.zeros(N)
    for _ in range(N):
        # Cycle kernels
        manifold = GeneralizedEllipse(target.mean, target.cov, target.pdf(x))
        y, a1 = zappa_step_acceptPC(x, manifold, p, A, m, tol, a_guess)
        x, a2 = kernel2(y.flatten(), **args2)
        samples, accept1[_], accept2[_] = np.vstack((samples, y, x)), a1, a2
    return samples[1:], accept1, accept2, time.time() - t