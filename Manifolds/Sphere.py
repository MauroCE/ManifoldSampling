import numpy as np
from Manifolds.Manifold import Manifold
import matplotlib.pyplot as plt

class Sphere(Manifold):
    def __init__(self, mu, r):
        """
        Class for a sphere. It collects functions and information that relates to a sphere.

        mu : Numpy Array
             Center of the sphere. Must be a 1D array of dimension (3, )
        r : Float
            Radius
        """
        super().__init__(m=1, d=2)
        self.mu = mu
        self.r = r


    def to_cartesian(self, theta_phi):
        """[θ, ϕ] --> [x, y, z]"""
        theta, phi = theta_phi[0], theta_phi[1]
        x = self.mu[0] + self.r * np.cos(phi) * np.sin(theta)
        y = self.mu[1] + self.r * np.sin(phi) * np.sin(theta)
        z = self.mu[2] + self.r * np.cos(theta)
        return np.array([x, y, z])

    def Q(self, xyz):
        """Q"""
        return (2*xyz - 2*self.mu).reshape(-1, self.m)
    
    def q(self, xyz):
        """Constraint function for the sphere"""
        return np.sum((xyz - self.mu)**2) - self.r**2

    