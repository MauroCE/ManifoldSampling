import numpy as np
from numpy.linalg import inv, solve
from scipy.stats import multivariate_normal as MVN


class HMC:
    """
    Basic HMC algorithm using Leapfrog integration and using a Euclidean-Gaussian kinetic energy. That is
    p ~ N(0, M) where M does not depend on q. This works for any target distribution.
    """
    def __init__(self, q0, n, M, T, epsilon):
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
        """
        # Store variables
        self.q0 = q0       # Initial position
        self.d = len(q0)   # Dimension of samples
        self.n = n         # number of Samples 
        self.M = M         # Covariance matrix
        self.Minv = np.linalg.inv(self.M)
        self.T = T
        self.epsilon = epsilon
        self.Ngrad = 0     # Number of gradient evaluations computed so far
        self.Ndens = 0     # Number of density evaluations computed so far
        
    @staticmethod
    def dVdq(self, q):
        """
        Computes the derivative of the potential energy with respect to the position, evaluated at q.
        
        q : Numpy Array
            Position at which we want to evaluate the derivative.
        """
        raise NotImplementedError

    @staticmethod
    def neg_log_target(self, q):
        """Negative Log Density of target distribution."""
        raise NotImplementedError
    
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
        self.Ngrad += 1

        # n - 1 full steps of both position and momentum
        for i in range(int(self.T / self.epsilon) - 1):
            q = q + self.epsilon * (self.Minv @ p)
            p = p - self.epsilon * self.dVdq(q)
            self.Ngrad += 1

        # Last full position step
        q = q + self.epsilon * (self.Minv @ p)
        # Final half-step 
        p = p - (self.epsilon / 2) * self.dVdq(q)
        self.Ngrad += 1

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
        samples = np.zeros((self.n + 1, self.d))
        samples[0] = self.q0

        # Store acceptances here
        acceptances = np.zeros(self.n)
        
        # Uniforms for MH correction
        logu = np.log(np.random.rand(self.n))
        
        # Store distributions (momentum distribution)
        momdis = MVN(mean=np.zeros(self.d), cov=self.M)
        H = lambda q, p : self.neg_log_target(q) - momdis.logpdf(p)

        # Sample momentum. Must have same dimension as q, i.e. 2D here
        ps = momdis.rvs(self.n).reshape(-1, self.d)     # (n, 2). 
        # Reshape(-1, 2) does nothing when n>1. For n=1 we make sure its (1, 2) rather than (2,) so that enumerate works

        # For every sample do leapfrog integration and MH correction
        for i, p in enumerate(ps):
            q = samples[i]
            q_prime, p_prime = self.leapfrog(q, p)
            if logu[i] <= H(q, p) - H(q_prime, p_prime):
                # Accept
                q = q_prime
                acceptances[i] = 1.0
            samples[i + 1] = q
        # Return all samples except for the first one
        return samples[1:], acceptances
        