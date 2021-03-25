import numpy as np
from scipy.stats import multivariate_normal
from utils import normalize
from numpy.linalg import cholesky, inv, eigh, cholesky, solve, det, norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.linalg import LinAlgError


class HugHop:

    def __init__(self, T, B, x0, q, logpi, grad_log_pi, l, k):
        """
        T : Float
            Total Integration time.
        B : Int
            Number of steps per iteration.
        x0 : Numpy Array
             Starting point of the algorithm. Should have shape (d, ).
        q : Callable
            Symmetric proposal for the velocity. Should have a rvs() method and a 
            logpdf() method. For instance could be multivariate_normal().
        logpi : Callable
                Log target density. E.g. for MVN would be multivariate_normal().logpf
        grad_log_pi : Callable
                      Gradient of log target density. E.g. for multivariate normal would be
                      lambda xy: - inv(Sigma) @ xy
        l : Float
            Lambda. Controls scaling parallel to the gradient.
        k : Float 
            We have mu^2 = k * l where mu^2 controls scaling perpendicular to gradient.
        """
        self.T = T
        self.B = B
        self.delta = self.T / self.B
        self.x0 = x0
        self.q = q
        self.logpi = logpi
        self.grad_log_pi = grad_log_pi
        self.l = l 
        self.k = k
        self.mu_sq = self.k * self.l
        self.mu = np.sqrt(self.mu_sq)
        self.l_sq = self.l**2
        # Find the dimensionality
        assert len(self.x0.shape) == 1, "x0 Must have shape (d, ). Found shape {}".format(self.x0.shape)
        self.d = len(self.x0)

    def HugKernel(self, x0):
        """
        Standard Hug Kernel. This uses no preconditioning. This is ONE STEP of the hug kernel.
        Returns a triplet (x, v, a) where x is the new sample, v is the velocity at the new sample
        and a is a binary flag indicating successful acceptance (a=1) or rejection (a=0).

        x0 : Numpy Array
             Point from which to do 1 step fo Hug. Basically the difference between self.x0 and
             x0 is that self.x0 is the starting point of the whole algorithm, while x0 is just the 
             starting point for this Hug.
        """
        # Draw velocity
        v0 = self.q.rvs()
        # Housekeeping
        v = v0
        x = x0
        # Acceptance ratio
        logu = np.log(np.random.rand())

        for _ in range(self.B):
            # Move
            x = x + self.delta*v/2 
            # Reflect
            g = self.grad_log_pi(x)
            ghat = g / norm(g)
            v = v - 2*(v @ ghat) * ghat
            # Move
            x = x + self.delta*v/2

        if logu <= self.logpi(x) + self.q.logpdf(v) - self.logpi(x0) - self.q.logpdf(v0):
            return (x, v, 1)   # 1 means accepted
        else:
            return (x0, v0, 0)  # 0 means rejected

    def HopKernel(self, x):
        """
        Standard Hop Kernel with no preconditioning. This is ONE STEP of the hop kernel.
        Returns a tuple (x, a) where x is the new sample, and a is a binary flag indicating 
        successful acceptance (a=1) or rejection (a=0).

        x : Numpy Array
            Point form which to perform a Hop.
        """       
        # For MH step
        logu = np.log(np.random.rand())
        
        # Gradient, its norm and nornmalized gradient
        gx = self.grad_log_pi(x)
        gx_norm = norm(gx)
        gxhat = gx / gx_norm
        
        # B^{1/2}
        B_sqrt = (self.mu*np.eye(self.d) + (self.l - self.mu)*np.outer(gxhat, gxhat)) / np.sqrt(1 + gx_norm**2)
        
        # Sample 
        v = multivariate_normal(mean=np.zeros(self.d), cov=np.eye(self.d)).rvs()
        
        # Proposal
        y = x + (B_sqrt @ v)
        
        # Compute gradient stuff at y
        gy = self.grad_log_pi(y)
        gy_norm = norm(gy)
        gyhat = gy / gy_norm
        
        # Accept-Reject
        logr = self.logpi(y) - self.logpi(x) + (self.d/2) * np.log((gy_norm**2) / (gx_norm**2))
        logr = logr - (1/(2*self.mu_sq)) * (norm(y-x)**2)*((gy_norm**2) - (gx_norm**2))
        logr = logr - 0.5*((1/self.l_sq) - (1/self.mu_sq)) * (((y - x) @ gy)**2 - ((y - x) @ gx)**2)
        if logu <= min(0, logr):
            # Accept
            return y, 1
        else:
            return x, 0

    def sample(self, N):
        """
        Alternates HugKernel and HopKernel N times.
        Returns a tripled (samples, acceptance_hug, acceptance_hop). Samples is 

        N : Int
            Number of Iterations.

        Return
        ------
        samples : Numpy Array
                  (N, d) array containing samples.
        acceptance_hug : Numpy Array
                         (N, ) array containing 0/1 flags with 1 meaning Hug succeded.
        acceptance_hop : Numpy Array
                         (N, ) array containing 0/1 flags with 1 meaning Hop succeded.
        """
        # Housekeeping
        x = self.x0
        samples = x
        acceptance_hug = np.zeros(N)
        acceptance_hop = np.zeros(N)

        for i in range(N):
            # Hug Kernel
            x_hug, v, ahug = self.HugKernel(x)
            # Hop Kernel
            x, ahop = self.HopKernel(x_hug)
            # Housekeeping
            samples = np.vstack((samples, x_hug))
            samples = np.vstack((samples, x))
            acceptance_hug[i] = ahug
            acceptance_hop[i] = ahop
        
        return samples, acceptance_hug, acceptance_hop
