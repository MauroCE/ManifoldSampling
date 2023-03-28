import math
import numpy as np
from numpy import zeros, eye, ones, log, exp, sqrt, diag, pi, vstack
from numpy.random import default_rng, randn
from numpy.linalg import norm
from scipy.stats import multivariate_normal as MVN
from scipy.optimize import fsolve
from warnings import catch_warnings, filterwarnings
from Manifolds.Manifold import Manifold


class GKManifold(Manifold):
    def __init__(self, ystar, n_chains=4):
        """Initiates class for the G and K manifold."""
        self.m = len(ystar)
        self.d = 4
        self.n = self.d + self.m
        self.ystar = ystar 
        self.n_chains = n_chains

    def q(self, ξ):
        """Constraint function."""
        with catch_warnings():
            filterwarnings('error')
            try:
                return (ξ[0] + ξ[1]*(1 + 0.8*(1 - exp(-ξ[2]*ξ[4:]))/(1 + exp(-ξ[2]*ξ[4:]))) * ((1 + ξ[4:]**2)**ξ[3])*ξ[4:]) - self.ystar
            except RuntimeWarning:
                raise ValueError("Constraint found Overflow warning.")

    def Q(self, ξ):
        """Transpose of the Jacobian of the constraint function."""
        return vstack((
            ones(len(ξ[4:])),
            (1 + 0.8 * (1 - exp(-ξ[2] * ξ[4:])) / (1 + exp(-ξ[2] * ξ[4:]))) * ((1 + ξ[4:]**2)**ξ[3]) * ξ[4:],
            8 * ξ[1] * (ξ[4:]**2) * ((1 + ξ[4:]**2)**ξ[3]) * exp(ξ[2]*ξ[4:]) / (5 * (1 + exp(ξ[2]*ξ[4:]))**2),
            ξ[1]*ξ[4:]*((1+ξ[4:]**2)**ξ[3])*(1 + 9*exp(ξ[2]*ξ[4:]))*log(1 + ξ[4:]**2) / (5*(1 + exp(ξ[2]*ξ[4:]))),
            diag(ξ[1]*((1+ξ[4:]**2)**(ξ[3]-1))*(((18*ξ[3] + 9)*(ξ[4:]**2) + 9)*exp(2*ξ[2]*ξ[4:]) + (8*ξ[2]*ξ[4:]**3 + (20*ξ[3] + 10)*ξ[4:]**2 + 8*ξ[2]*ξ[4:] + 10)*exp(ξ[2]*ξ[4:]) + (2*ξ[3] + 1)*ξ[4:]**2 + 1) / (5*(1 + exp(ξ[2]*ξ[4:]))**2))
        ))

    def compute_J(self, ξ):
        """Computes Jacobian of the constraint function."""
        return self.Q(ξ).T

    def logη(self, ξ):
        """Density on Manifold wrt Hausdorff measure."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        try:
            J = self.compute_J(ξ)
            logprior = -ξ@ξ/2
            correction_term  = - math.prod(np.linalg.slogdet(J@J.T))/2 
            return  logprior + correction_term
        except ValueError as e:
            return -np.inf

    def log_normal_kernel(self, ξ, ϵ):
        """Log-normal kernel. NOTE: This is an approximation to the identity."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        u = self.q(ξ)
        return -self.n*log(2*np.pi)/2 - self.n*log(ξ) - u.dot(u)/(2*ϵ**2)

    def generate_logpi(self, ϵ):
        """Generates ABC posterior using a certain epsilon value. Uses a Gaussian kernel. """
        logηϵ = lambda ξ: self.log_normal_kernel(ξ, ϵ) - ξ@ξ/2
        return logηϵ

    def find_point_on_manifold(self, maxiter=2000, tol=1e-14, random_z_guess=False):
        """Finds a point on the Manifold."""
        z_guess = randn(self.m) if random_z_guess else zeros(self.m)
        i = 0
        with catch_warnings():
            filterwarnings('error')
            while i <= maxiter:
                i += 1
                try: 
                    # Sample theta from the prior
                    
                    u1_init  = randn(self.d)*0.1 - 4
                    function = lambda u2: self.q(np.concatenate((u1_init, u2)))
                    fprime   = lambda u2: self.J(np.concatenate((u1_init, u2)))[:, self.d:]
                    u2_found = fsolve(function, u2_guess, xtol=tol, fprime=fprime)
                    u_found = np.concatenate((u1_init, u2_found))
                    return u_found
                except RuntimeWarning:
                    continue
        raise ValueError("Couldn't find a point, try again.")

    def find_init_points_for_each_chain(self, theta_true=True, random_z_guess=False, tol=1e-14, maxiter=5000):
        """Finds `n_chains` initial points on the manifold.

        Args:
            theta_true (boool, optional): Whether to use theta0 that generated the data or sample it at random.
            random_z_guess (bool, optional): Whether to generate the initial z guess at random or as a zero vector. Defaults to False.
            tol (float, optional): tolerance for fsolve. Defaults to 1e-14.
            maxiter (int, optional): Maximum number of iterations for optimization procedure. Defaults to 5000.

        Returns:
            ndarray: array having dimension (n_chains, n), containing each point on a row.
        """
        ξ0s = zeros((self.n_chains, self.n))
        for i in range(self.n_chains):
            if theta_true:
                ξ0s[i, :] = self.find_point_on_manifold_given_u1true(maxiter=maxiter, tol=tol, random_z_guess=random_z_guess)
            else:
                ξ0s[i, :] = self.find_point_on_manifold(maxiter=maxiter, tol=tol, random_z_guess=random_z_guess)
        self.ξ0s = ξ0s 
        return self.ξ0s

    def is_on_manifold(self, ξ, tol=1e-14):
        """Checks if a point is on the manifold."""
        return max(abs(self.q(ξ))) <= tol


