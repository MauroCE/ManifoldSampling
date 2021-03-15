import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal


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
            q = q + self.epsilon * p
            p = p - self.epsilon * self.dVdq(q)

        # Last full position step
        q = q + self.epsilon * p
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
        return samples
        