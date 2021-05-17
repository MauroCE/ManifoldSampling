import numpy as np
from numpy.linalg import svd, eigh, inv
from math import log, sqrt
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time 

####################################################################################################################################################################
########################################### R O T A T E D      E L L I P S E
####################################################################################################################################################################
class Manifold:
    def __init__(self, m, d):
        """
        Generic Manifold Class.

        m : Int
            Number of constraints & Co-dimension of the manifold. (e.g. 1 for Torus/Sphere)
        d : Int
            Dimension of the manifold. (e.g. 2 for Torus/Sphere)
        """
        self.m = m
        self.d = d

    def tangent_basis(self, Q):
        """
        Computes a tangent basis from the Q matrix (the transpose of the Jacobian matrix).

        Q : Numpy Array
            2D Numpy array of dimension (m + d, m) containing gradients of the constraints as columns.
        returns : Matrix containing basis of tangent space as its columns.
        """
        assert Q.shape == (self.m + self.d, self.m), "Q must have shape ({}, {}) but found shape {}".format(self.m+self.d, self.m, Q.shape)
        return svd(Q)[0][:, self.m:]

    def get_dimension(self):
        """
        Returns dimension of the manifold d
        """
        return self.d
    
    def get_codimension(self):
        """
        Returns co-dimension of the manifold d
        """
        return self.m

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

    def Q_old(self, xy):
        """
        Transpose of the Jacobian.
        """
        xy = xy - self.mu
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
        v1_cc_v2 = int(v1[0]*v2[1] < v2[0]*v1[1])
        #v1_cc_v2 = int((v2[1] + v1[0] == 0))
        # Remember if v1 cc v2 then we use v2, not v1
        theta = np.arctan2(*(v1_cc_v2*v2 + (1 - v1_cc_v2)*v1)[::-1])
        # Compute a^2 and b^2
        a_sq = self.gamma * vals[v1_cc_v2]
        b_sq = self.gamma * vals[1 - v1_cc_v2]
        return a_sq, b_sq, theta    

    def J(self, z):
      """
      Computes Jacobian of the reparametrization. See notes.
      """
      lambda_a = self.a_sq / self.gamma
      lambda_b = self.b_sq / self.gamma
      return np.sqrt(lambda_a * lambda_b) / z

    def peri(self):
      """ Computes perimeter of ellipse using Ramanujan's formula. """
      return np.pi * (3*(self.a+self.b) - sqrt((3*self.a + self.b) * (self.a + self.b*3)))

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

####################################################################################################################################################################
########################################### H M C
####################################################################################################################################################################

class GaussianTargetHMC:
    """
    Basic HMC algorithm using Leapfrog integration and using a Euclidean-Gaussian kinetic energy. That is
    p ~ N(0, M) where M does not depend on q. Importantly, this only works for a Gaussian target distribution 
    with covariance matrix Sigma and mean mu.
    """
    def __init__(self, q0, n, M, T, epsilon, Sigma, mu):
        """
        q0 : Numpy Array
             Starting position. 
             
        n : Int
            Number of samples we want to get from the target distribution.
            
        M : Numpy Array
            Covariance matrix for the conditional momentum distribution p(p|q).
            
        T : Float
            Total integration time of a trajectory for the Leapfrog integrator.
            
        epsilon : Float
                  Step size for Leapfrog integrator.
                  
        Sigma : Numpy Array
                Covariance matrix of the target distribution.
                
        mu : Numpy Array
             Mean of the target distribution.
        """
        # Store variables
        self.q0 = q0
        self.n = n
        self.M = M
        self.Minv = np.linalg.inv(self.M)
        self.T = T
        self.epsilon = epsilon
        self.Sigma = Sigma
        self.mu = mu
        
    def dVdq(self, q):
        """
        Computes the derivative of the potential energy with respect to the position, evaluated at q.
        
        q : Numpy Array
            Position at which we want to evaluate the derivative.
        """
        return inv(self.Sigma) @ (q - self.mu)
    
    def leapfrog(self, q, p):
        """
        Integrates using the Leapfrog integrator.
        
        q : Numpy Array
            Initial position q0.
        p : Numpy Array
            Initial momentum p0.
        """
        # First momentum half-step 
        p = p - (self.epsilon / 2) * self.dVdq(q)

        # n - 1 full steps of both position and momentum
        for i in range(int(self.T / self.epsilon) - 1):
            q = q + self.epsilon * (self.Minv @ p)
            p = p - self.epsilon * self.dVdq(q)

        # Last full position step
        q = q + self.epsilon * (self.Minv @ p)
        # Final half-step 
        p = p - (self.epsilon / 2) * self.dVdq(q)

        # Return momentum flipped for reversibility
        return q, -p
    
    def sample(self):
        """
        Samples from the model using HMC.

        Returns
        -------

        A Numpy Array of size (n + 1, 2) containing q0 at index 0 and then the n samples.
        """
        # Store all samples here
        samples = np.zeros((self.n + 1, 2))
        samples[0] = self.q0
        
        # Uniforms for MH correction
        logu = np.log(np.random.rand(self.n))
        
        # Store distributions (target and momentum distribution)
        target = multivariate_normal(mean=self.mu, cov=self.Sigma)
        momdis = multivariate_normal(mean=np.zeros(2), cov=self.M)
        H = lambda q, p : -target.logpdf(q) - momdis.logpdf(p)

        # Sample momentum. Must have same dimension as q, i.e. 2D here
        ps = momdis.rvs(self.n).reshape(-1, 2)     # (n, 2). 
        # Reshape(-1, 2) does nothing when n>1. For n=1 we make sure its (1, 2) rather than (2,) so that enumerate works

        # For every sample do leapfrog integration and MH correction
        for i, p in enumerate(ps):
            q = samples[i]
            q_prime, p_prime = self.leapfrog(q, p)
            if logu[i] <= H(q, p) - H(q_prime, p_prime):
                # Accept
                q = q_prime
            samples[i + 1] = q
        # Return all samples except for the first one
        return samples[1:]
        
####################################################################################################################################################################
########################################### S E T T I N G S 
####################################################################################################################################################################

mu = np.zeros(2) 
Sigma = np.array([[1.0, 0.99], [0.99, 2.0]])
target = multivariate_normal(mu, Sigma)
n = 2000000                                    
T = 5                                        
epsilon = 0.1                                
M = np.eye(2) 

####################################################################################################################################################################
########################################### R U N
####################################################################################################################################################################

q0 = target.rvs()
hmc = GaussianTargetHMC(q0, n, M, T, epsilon, Sigma, mu)
start_time = time.time()
samples = hmc.sample()
end_time = time.time()
np.save("hmc_samples", samples)
np.save("hmc_time", end_time - start_time)

