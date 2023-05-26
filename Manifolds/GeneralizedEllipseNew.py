import numpy as np
from numpy import log
from numpy import pi
from numpy import array
from numpy.linalg import det, inv, solve, norm
from numpy.random import randn, rand, randint, default_rng
from scipy.optimize import fsolve
from scipy.stats import multivariate_normal as MVN
from scipy.stats import norm as ndist
from scipy.stats import uniform as udist
from Manifolds.Manifold import Manifold


class GeneralizedEllipse(Manifold):
    def __init__(self, mu, Sigma, z, prior='normal', prior_loc=0.0, prior_scale=1.0, kernel='uniform'):
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
        self.prior = prior
        self.kernel = kernel
        super().__init__(m=1, d=(self.n-1))

        # choose prior function
        if self.prior == 'normal':
            def sample_prior_normal(n, seed=None):
                if seed is None:
                    seed = randint(low=1000, high=9999)
                rng = default_rng(seed=seed)
                return rng.normal(loc=prior_loc, scale=prior_scale, size=(n, len(mu)))
            self.sample_prior = sample_prior_normal
            self.logprior     = lambda ξ: ndist.logpdf(ξ, loc=prior_loc, scale=prior_scale).sum()
        elif self.prior == 'uniform':
            def sample_prior_uniform(n, seed=None):
                if seed is None:
                    seed = randint(low=1000, high=9999)
                rng = default_rng(seed=seed)
                ##### We want the prior to be U(-c, c) for some constant c. This means that if c=10 one wants
                #### U(-10, 10) and not U(0, 10). To get this with scipy we basically need to shift and scale
                #### loc=-c, scale=2*c will do the job. This is why its different for rng.uniform and
                ### for udist 
                return rng.uniform(low=-prior_scale, high=prior_scale, size=(n, len(mu)))
            self.sample_prior = sample_prior_uniform
            self.logprior     = lambda ξ: udist.logpdf(ξ, loc=(-prior_scale), scale=2*prior_scale).sum()
        else:
            raise ValueError("Invalid prior specification.")

    def generate_logηε(self, ε):
        """Generates the filamentary distribution."""
        if self.kernel == 'uniform':
            def logηε(ξ):
                with np.errstate(divide='ignore'):
                    return self.logprior(ξ) + log(float(norm(self.q(ξ)) <= ε)) - log(ε)
            return logηε
        elif self.kernel == 'normal':
            def logηε(ξ):
                u = self.q(ξ)
                return self.logprior(ξ) - (norm(u)**2)/(2*ε**2) - log(ε) - log(2*np.pi)/2
            return logηε

    def q(self, xyz):
        """Constraint function for the contour of MVN"""
        return self.MVN.logpdf(xyz) - log(self.z)

    def Q(self, xyz):
        """Q"""
        return (self.Sinv @ (xyz - self.mu)).reshape(-1, self.m)

    def grad(self, xyz):
        """Gradient function. Not sure why but it's half of Q."""
        return -solve(self.S, xyz - self.mu)

    def fullJacobian(self, xyz):
        """This is just so that all manifold classes have the same interface."""
        return self.grad(xyz)

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
