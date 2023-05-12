import numpy as np
from numpy.linalg import norm


class Constraint:
    """Constraint functions for manifold sampling."""

    def __init__(self, function, input_dimension, output_dimension, constraint_value):
        """Represents the constraint for a certain approximate manifold sampling problem."""
        self.f = function
        self.n = input_dimension
        self.m = output_dimension
        self.y = constraint_value

    def __call__(self, x):
        """Evaluates the constraint function."""
        return self.f(x) - self.y


class GaussianKernel:
    """Gaussian kernel."""

    def __init__(self, f, epsilon, log=True):
        self.f = f 
        self.n = f.n
        self.m = f.m
        self.y = f.y
        self.epsilon = epsilon
        self.log = log


        def log_gaussian_kernel(x):
            return - (self.m/2)*np.log(2*np.pi) - self.m*np.log(epsilon) - (norm(f(x))**2)/(2*epsilon**2)
        
        def gaussian_kernel(x):
            return np.exp(-norm(f(x))**2 / (2*epsilon**2)) / ((2*np.pi)**(self.m/2) * (epsilon**self.m))

        if log:
            self.evaluate_kernel = log_gaussian_kernel
        else:
            self.evaluate_kernel = gaussian_kernel

    def __call__(self, x):
        return self.evaluate_kernel(x)