import numpy as np
from numpy import log
from numpy import pi
from numpy import array
from numpy.linalg import det, inv, solve
from scipy.optimize import fsolve
from scipy.stats import multivariate_normal as MVN
from Manifolds.Manifold import Manifold


class GeneralizedEllipse(Manifold):
    def __init__(self, mu, Sigma, z):
        """
        Class for a general MVN ellipse.

        mu : Numpy Array
             Center of the sphere. Must be a 1D array of dimension (3, )
        r : Float
            Radius
        """
        self.n = len(mu)    # Dimension of the ambient space
        # Store MVN parameters
        self.z = z
        self.mu = mu
        self.S = Sigma
        self.Sinv = inv(Sigma)
        self.MVN = MVN(self.mu, self.S)
        super().__init__(m=1, d=(self.n-1))

    def q(self, xyz):
        """Constraint function for the contour of MVN"""
        return self.MVN.logpdf(xyz) - log(self.z)

    def Q(self, xyz):
        """Q"""
        return (2 * self.Sinv @ (xyz - self.mu)).reshape(-1, self.m)

    def sample(self, advanced=False, maxiter=10000):
        """Samples from the contour by first sampling a point from the original
        MVN and then it rescales it until it is on the correct contour. This should
        work since the MVN is spherically symmetric."""
        if not advanced:
            start = self.MVN.rvs()   # Get initial MVN sample
        else:
            start = self.find_point_near_manifold(maxiter=maxiter)
        objective = lambda coef: self.MVN.pdf(coef*start) - self.z  # Objective function checks closeness to z
        optimal_coef = fsolve(objective, 1.0) # Find coefficient so that optimal_coef*start is on contour
        return start * optimal_coef
    
    def find_point_near_manifold(self, maxiter=10000):
        """Finds a point near the manifold"""
        samples = self.MVN.rvs(maxiter)
        index = np.argmin(abs(self.MVN.pdf(samples) - self.z))
        return samples[index, :]
    
