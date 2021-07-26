import numpy as np
from numpy import log
from numpy import pi
from numpy.linalg import det, inv, solve
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
        self.mu = mu
        self.logdetS = log(det(self.S))
        # Compute gamma (RHS) 
        self.gamma = -self.n * log(2*pi) - self.logdetS -2 * log(z)
        super().__init__(m=1, d=(self.n-1))

    def q(self, xyz):
        """Constraint function for the contour of MVN"""
        return (xyz - self.mu) @ (self.Sinv @ (xyz - self.mu)) - self.gamma

    def Q(self, xyz):
        """Q"""
        return (2 * self.Sinv @ (xyz - self.mu)).reshape(-1, self.m)
    

class GeneralizedEllipsePC(Manifold):
    def __init__(self, mu, Sigma, z, A):
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
        self.mu = mu
        self.logdetS = log(det(self.S))
        # Compute gamma (RHS) 
        self.gamma = -self.n * log(2*pi) - self.logdetS -2 * log(z)
        super().__init__(m=1, d=(self.n-1))

    def q(self, xyz):
        """Constraint function for the contour of MVN"""
        return (xyz - self.mu) @ (self.Sinv @ (xyz - self.mu)) - self.gamma

    def Q(self, xyz):
        """Q"""
        return (2 * self.Sinv @ (xyz - self.mu)).reshape(-1, self.m)
    
