import numpy as np
from numpy import zeros, log, eye, vstack, arange, repeat, hstack
from numpy.random import rand, default_rng, randint
from scipy.stats import multivariate_normal as MVN

def RWM(x0, s, N, logpi, seed=None):
    """Simple RWM function with proposal N(x, s*I)"""
    if seed is None:
        seed = randint(low=1000, high=9999)
    rng = default_rng(seed=seed)
    samples = x = x0                                   # Accepted samples will be stored here
    acceptances, logpx, d = zeros(N), logpi(x), len(x) # Accepted (=1), log(pi(x)), dimensionality
    logu = log(rng.uniform(size=N))                                # Used for Accept/Reject step
    # q = MVN(zeros(d), s * eye(d))                      # Proposal distribution is MVN with s * I covariance matrix.

    for i in range(N):
        y = x + rng.normal(loc=0.0, scale=s, size=d)#q.rvs()                # Sample candidate
        logpy = logpi(y)               # Compute its log density
        if logu[i] <= logpy - logpx:
            x, logpx, acceptances[i] = y, logpy, 1 # Accept!
        samples = vstack((samples, x)) # Add sample to storage
    return samples[1:], acceptances


def generate_RWMIntegrator(B, δ, metropolised=False):
    """This function is used mostly in SMC and Markov Snippets. The interface
    is a bit weird just because we want it to look like the interface for THUG.
    Given a step size δ and a number of steps B, it basically returns a function/class
    that can be applied to a point z = [x, v] and returns B points on a straight line
    starting from x and with direction v.
    The parameter metropolised tells us whether to return the whole trajectory (False)
    or the first and last point only."""
    def RWMIntegrator(z0, B, δ, metropolised=False):
        """Random Walk integrator. Given z0 = [x0, v0] it constructs a RW trajectory
        with B steps of step-size δ. This is a deterministic trajectory, and since
        RWM does not use gradients, this corresponds to B+1 points in a straight line
        starting from x0 and in the direction of v0."""
        x0, v0 = z0[:len(z0)//2], z0[len(z0)//2:]
        bs  = arange(B+1).reshape(-1, 1) # 0, 1, ..., B
        xbs = x0 + δ*bs*v0     # move them by b*δ
        vbs = repeat(v0.reshape(1, -1), repeats=B+1, axis=0)
        zbs = hstack((xbs, vbs))
        if not metropolised:
            return zbs
        else:
            return zbs[[0, -1], :]

    class RWMIntegratorClass:
        def __init__(self, B, δ, metropolised=False):
            self.B   = B
            self.δ   = δ
            self.metropolised = metropolised

        def __repr__(self):
            if not self.metropolised:
                return "RWM Integrator with B = {} and δ = {:.6f}".format(self.B, self.δ)
            else:
                return "Metropolised RWM Integrator with B = {} and δ = {:.6f}".format(self.B, self.δ)

        def __call__(self, z):
            integrator = lambda z: RWMIntegrator(z, self.B, self.δ, metropolised=self.metropolised)
            return integrator(z)
    return RWMIntegratorClass(B, δ, metropolised=metropolised)


def RWM_Cov(x0, Sigma, N, logpi):
    """RWM using a user-specified coviariance matrix Sigma."""
    samples = x = x0                                   # Accepted samples will be stored here
    acceptances, logpx, d = zeros(N), logpi(x), len(x) # Accepted (=1), log(pi(x)), dimensionality
    logu = log(rand(N))                                # Used for Accept/Reject step
    q = MVN(zeros(d), Sigma)                           # Proposal distribution is MVN with s * I covariance matrix.

    for i in range(N):
        y = x + q.rvs()                # Sample candidate
        logpy = logpi(y)               # Compute its log density
        if logu[i] <= logpy - logpx:
            x, logpx, acceptances[i] = y, logpy, 1 # Accept!
        samples = vstack((samples, x)) # Add sample to storage
    return samples[1:], acceptances
