import numpy as np
from numpy.random import rand, randn
from numpy import exp, inf
from numpy.linalg import cholesky, norm
from scipy.optimize import solve



class ConstrainedHMC:
    def __init__(self, dt, Ng, Ns, epsilon):
        """Constrained HMC sampler."""
        self.dt = dt 
        self.Ng = Ng
        self.Ns = Ns 
        self.epsilon = epsilon

    @staticmethod
    def constraint_function(x):
        """Takes a position x and returns the value of the constraint funciton.
        C-HMC targets the zero level-set of this function."""
        raise NotImplementedError("Constraint function not provided.")

    @staticmethod
    def gradient_constraint(x):
        """Gradient of constraint function."""
        raise NotImplementedError("Gradient fo Constraint function not provided.")

    @staticmethod
    def V(x):
        """Potential energy"""
        raise NotImplementedError("Potential Function not provided.")

    @staticmethod
    def dVdx(x):
        raise NotImplementedError("Gradient of Potential Energy not provided.")

    @staticmethod
    def K(x, v):
        """Kinetic energy"""
        raise NotImplementedError("Kinetic Function not provided.")

    def H(self, x, v):
        """Hamiltonian funciton."""
        return self.V(x) + self.K(x, v)

    def project_momentum(self, v, J, L):
        """Function to project momentum back to tangent space."""
        return v - J.T @ solve(L.T, solve(L, J @ v))

    def project_position(self, x, J, L):
        c = self.constraint_function(x)
        while norm(c, inf) > self.epsilon:
            x = x - J.T @ solve(L.T, solve(L, c))
            c = self.constraint_function(x)
        return x

    def simulate_geodesic(self, x, v, J, L):
        """Simulates geodesic on cotangent bundle."""
        for _ in range(self.Ng):
            x_tilde = x + self.dt * v / self.Ng
            x_prime = self.project_position(x_tilde, J, L)
            J = self.gradient_constraint(x_prime)
            L = cholesky(J@J.T)
            v_tilde = self.Ng * (x_prime - x) / self.dt 
            v = self.project_momentum(v_tilde, J, L)
            x = x_prime
        return x, v, J, L

    def simulate_dynamic(self, x, v, J, L):
        """Simulates C-HMC dynamic."""
        v = v + self.dt * self.dVdx(x) / 2
        v = self.project_momentum(v)
        x, v, J, L = self.simulate_geodesic(x, v, J, L)
        for _ in range(self.Ns - 1):
            v_tilde = v + self.dt * self.dVdx(x)
            v = self.project_momentum(v_tilde, J, L)
            x, v, J, L = self.simulate_geodesic(x, v, J, L)
        v_tilde = v + self.dt * self.dVdx(x) / 2
        v = self.project_momentum(v_tilde, J, L)
        return x, v, J, L

    def sample(self):
        """TODO"""
        x, v, J, P = np.zeros(4)
        d = len(x)
        # Simulate dynamic
        xp, vp, Jp, Lp = self.simulate_dynamic(x, v, J, P)
        if rand() < exp(self.H(x, v) - self.H(xp, vp)):
            x, v, J, L = xp, vp, Jp, Lp  # Accepted
        n = randn(d)
        p = self.project_momentum(n, J, L)
        


    
