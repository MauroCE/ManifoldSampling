import numpy as np
from numpy.linalg import eigh, inv
from Manifolds.Manifold import Manifold
import matplotlib.pyplot as plt
from math import log



class RotatedEllipse(Manifold):
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
        # Store calculations
        self.ct = np.cos(self.theta)
        self.st = np.sin(self.theta)
        self.ctmst = np.array([self.ct, -self.st])   # (cos(theta), -sin(theta))
        self.stct = np.array([self.st, self.ct]) # (sin(theta), cos(theta))
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
        Transpose of the Jacobian.
        """
        dxq = (2*np.dot(xy, self.ctmst)*self.ct/self.a_sq) + (2*np.dot(xy, self.stct)*self.st/self.b_sq)
        dyq = -(2*np.dot(xy, self.ctmst)*self.st/self.a_sq) + (2*np.dot(xy, self.stct)*self.ct/self.b_sq)
        return np.array([dxq, dyq]).reshape(-1, self.m)
        
    
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
        denom = 4*(np.pi**2)*sx2*sy2*(1 - (rho**2))*(self.z**2)
        return rho, sx2, sy2, np.log(1 / denom)

    def _find_ab_theta(self):
        """
        Same as _find_ab_theta_old but more succint.
        """
        # Eigendecomposition of Sigma
        vals, P = eigh(self.S)
        v1, v2 = P[:, 0], P[:, 1]
        # Find out which one is counter-clockwise (cc). Here v1_cc_v2 stands for v1 counter-clockwise to v2
        v1_cc_v2 = int((v2[1] + v1[0] == 0))
        # Remember if v1 cc v2 then we use v2, not v1
        theta = np.arctan2(*(v1_cc_v2*v2 + (1 - v1_cc_v2)*v1)[::-1])
        # Compute a^2 and b^2
        a_sq = self.gamma * vals[v1_cc_v2]
        b_sq = self.gamma * vals[1 - v1_cc_v2]
        return a_sq, b_sq, theta    

    def _find_ab_theta_old(self):
        """
        This function proceeds as follows:

        - Takes the equation (x - \mu)^\top \Sigma^{-1} (x - \mu) = \gamma and divides both
          sides by \gamma. This means we can absorb (1/ \gamma) into \Sigma^{-1} and therefore
          we can use gamma*Sigma rather than Sigma. This gives us the equation of an ellipse.
        - For this reason, we compute the eigendecomposition of gamma*Sigma and grab its two eigenvectors
          v1 and v2 corresponding to eigenvalues values[0] and values[1] where values[0] < values[1].
        - To find \theta, it computes the dot product of v1 with e1 and v2 with e1 where e1 = (1, 0).
          This dot product is equal to cos(theta) and we use geometric arguments (i.e. sign of y component)
          to adjust this angle. Then theta is chosen to be the smallest angle because the bigger one will
          simply be theta + pi/2. 
        - We are also careful to grab the correspoding eigenvalues. That is, if v1 is the one with the smallest angle
          then it corresponds to e1 rotated and its corresponding value (values[0]) will be a^2. If v1 instead is the largest one,
          then it correspondst to e2 rotated and its corresponding value (values[0]) will be b^2.
        - Finally, to compute a^2 and b^2 we simply take the reciprocal of the values.

        This function then returns a^2, b^2, theta.
        """
        # Eigendecomposition. Find eigenvectors v1, v2 and eigenvalues
        values, P = eigh(inv(self.gamma*self.S))    # Values are in ASCENDING ORDER!!!
        v1, v2 = P[:, 0], P[:,1]
        # Dot product with standard basis vector to find angle of rotation
        e1 = np.array([1, 0])  # Standard Basis vector
        angle_v1e1 = (2*np.pi + np.sign(v1[1])*np.arccos(np.dot(v1, e1))) % (2*np.pi) # v1, e1
        angle_v2e1 = (2*np.pi + np.sign(v2[1])*np.arccos(np.dot(v2, e1))) % (2*np.pi) # v2, e1
        # Choose the minimum angle (other will be + 90°)
        angles = np.array([angle_v1e1, angle_v2e1])                                   # Together
        x_axis_ix = np.argmin(angles)                          # minimum
        theta = angles[x_axis_ix]
        # Be careful about the ordering. Recall values are in ascending order and that 
        # a corresponds to x-axis and b to y-axis.
        a_sq = (1 / values)[x_axis_ix]
        b_sq = (1 / values)[1 - x_axis_ix]
        return a_sq, b_sq, theta
