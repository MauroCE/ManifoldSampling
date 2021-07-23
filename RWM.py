import numpy as np
from numpy import zeros, log, eye, vstack
from numpy.random import rand
from scipy.stats import multivariate_normal as MVN

def RWM(x0, s, N, logpi):
    """Simple RWM function with proposal N(x, s*I)"""
    samples = x = x0                                   # Accepted samples will be stored here
    acceptances, logpx, d = zeros(N), logpi(x), len(x) # Accepted (=1), log(pi(x)), dimensionality
    logu = log(rand(N))                                # Used for Accept/Reject step
    q = MVN(zeros(d), s * eye(d))                      # Proposal distribution is MVN with s * I covariance matrix.

    for i in range(N):
        y = x + q.rvs()                # Sample candidate
        logpy = logpi(y)               # Compute its log density
        if logu[i] <= logpy - logpx:
            x, logpx, acceptances[i] = y, logpy, 1 # Accept! 
        samples = vstack((samples, x)) # Add sample to storage
    return samples[1:], acceptances


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