import numpy as np
from numpy.linalg import eigh
from Manifolds.Manifold import Manifold
import matplotlib.pyplot as plt
from math import log


class Ellipse(Manifold):
    def __init__(self, mu, a, b, theta):
        """
        Class for ellipse. If you want a circle of radius r > 0 simply set a=b=r.

        mu : Numpy Array
             Center of the Ellipse. Should be a 1D array of shape (2, ).
        a : Float
            Semi-mejor axis. Basically it is used in (x - mu[0])^2 / a^2
        b : Float
            Semi-minor axis. Basically it is used in (y - mu[1])^2 / b^2
        """
        super().__init__(m=1, d=1)
        self.mu = mu
        self.a = a
        self.b = b
        self.ab_sq = np.array([self.a**2, self.b**2])
        self.theta

    def to_cartesian(self, theta):
        """
        Transforms polar coordinates to cartesian.
        """
        # Define a function to compute the radius
        def r(t):
            # https://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center
            return (self.a * self.b) / np.sqrt((self.b*np.cos(t))**2 + (self.a*np.sin(t))**2)
        x = self.mu[0] + r(theta)*np.cos(theta)
        y = self.mu[1] + r(theta)*np.sin(theta)
        return np.array([x, y])
        # x = self.mu[0] + self.a * np.cos(theta)
        # y = self.mu[1] + self.b * np.sin(theta)
        # return np.array([x, y])

    def Q(self, xy):
        """
        Computes the transpose of the Jacobian.
        """
        return (2*(xy - self.mu) / self.ab_sq).reshape(-1, self.m)

    def q(self, xy):
        """
        Constraint defininig the manifold.

        xy : Numpy Array
             Point for which we want to compute the constraint. This should be Numpy Array with shape
             (2, ).
        """
        #return np.sum(((xy - self.mu)**2 / self.ab_sq)) - 1
        return ((xy[0] - self.mu[0]) / self.a)**2 + ((xy[1] - self.mu[1]) / self.b)**2 - 1


