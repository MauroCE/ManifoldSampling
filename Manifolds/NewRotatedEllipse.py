import numpy as np
from numpy.linalg import eigh, inv
from Manifolds.Manifold import Manifold
import matplotlib.pyplot as plt
from math import log, sqrt



class NewRotatedEllipse(Manifold):
    def __init__(self, mu, Sigma, z):
        """
        Rotated ellipse.
        """
        # Store MVN parameters
        self.z = z
        self.mu = mu
        self.S = Sigma
        self.rho, self.sx2, self.sy2, self.gamma = self._find_rho_variances_gamma()
        # Store Ellipse parameters
        self.a_sq, self.b_sq, self.theta = self._find_ab_theta()
        self.a = np.sqrt(self.a_sq)
        self.b = np.sqrt(self.b_sq)
        self.ab_sq = np.array([self.a_sq, self.b_sq])
        # Store calculations
        self.ct = np.cos(self.theta)
        self.st = np.sin(self.theta)
        self.ctmst = np.array([self.ct, -self.st])   # (cos(theta), -sin(theta))
        self.stct = np.array([self.st, self.ct]) # (sin(theta), cos(theta))
        # Counter-clockwise Rotation matrix
        self.R = np.array([[self.ct, -self.st], 
                           [self.st, self.ct]])
        # Clockwise Rotation matrix
        self.Rp = np.array([[self.ct, self.st], 
                            [-self.st, self.ct]])
        super().__init__(m=1, d=1)

    def to_cartesian(self, t):
        """
        Given an angle t, it computes a point in cartesian coordinates on the ellipse.
        Notice that t is NOT the angle wrt to the x-axis, but the angle relative to the rotated ellipse.
        """
        x = self.a * np.cos(t) * self.ct - self.b * np.sin(t) * self.st
        y = self.a * np.cos(t) * self.st + self.b * np.sin(t) * self.ct
        return np.array([x, y])

    def q(self, xy):
        """
        Constraint defining the manifold. Importantly, notice how the signs + and -
        are the opposite of the ones in wikipedia!
        """
        xc, yc = xy - self.mu
        xx = (xc*self.ct + yc*self.st)**2 / self.a_sq
        yy = (xc*self.st - yc*self.ct)**2 / self.b_sq
        return xx + yy -1

    def Q(self, xy):
        """
        New version of the gradient.
        """
        # Center the points and un-rotate them
        xy = self.Rp @ (xy - self.mu)
        return (self.R @ (2*xy / self.ab_sq)).reshape(-1, self.m)
        #return (self.R @ ((2 * xy) / self.ab_sq)).reshape(-1, self.m)

    def _find_rho_variances_gamma(self):
        """
        Returns:

        - rho : correlation between x and y
        - sx2 : the variance for x
        - sy2 : the variance for y
        - gamma : I have denoted gamma myself but basically it is what is left on the other side of the
                  contour equation once you have reduced it to a quadratic form 
                  (x - \mu)^\top \Sigma^{-1} (x - \mu) = \gamma
        """
        sx2 = self.S[0, 0] 
        sy2 = self.S[1, 1]
        rho = self.S[1, 0] / np.sqrt(sx2 * sy2)
        return rho, sx2, sy2, self.z**2

    def _find_ab_theta(self):
        """
        Same as _find_ab_theta_old but more succint.
        """
        # Eigendecomposition of Sigma
        vals, P = eigh(self.S)
        v1, v2 = P[:, 0], P[:, 1]
        # Find out which one is counter-clockwise (cc). Here v1_cc_v2 stands for v1 counter-clockwise to v2
        v1_cc_v2 = int(v1[0]*v2[1] < v2[0]*v1[1])
        #v1_cc_v2 = int((v2[1] + v1[0] == 0))
        # Remember if v1 cc v2 then we use v2, not v1
        theta = np.arctan2(*(v1_cc_v2*v2 + (1 - v1_cc_v2)*v1)[::-1])
        # Compute a^2 and b^2
        a_sq = self.gamma * vals[v1_cc_v2]
        b_sq = self.gamma * vals[1 - v1_cc_v2]
        return a_sq, b_sq, theta    

    def peri(self):
      """ Computes perimeter of ellipse using Ramanujan's formula. """
      return np.pi * (3*(self.a+self.b) - sqrt((3*self.a + self.b) * (self.a + self.b*3)))

