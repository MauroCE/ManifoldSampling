from numpy.linalg import svd
from scipy.optimize import root
from jax import jacfwd,jit

class Manifold:
    def __init__(self, m, d, q):
        """
        Generic Manifold Class.

        m : Int
            Number of constraints & Co-dimension of the manifold. (e.g. 1 for Torus/Sphere)
        d : Int
            Dimension of the manifold. (e.g. 2 for Torus/Sphere)
        """
        self.m = m
        self.d = d
        self.q=q
        self.Q=jit(jacfwd(self.q))

    def tangent_basis(self,x):
        """
        Computes a tangent basis from the Q matrix (the transpose of the Jacobian matrix).

        Q : Numpy Array
            2D Numpy array of dimension (m + d, m) containing gradients of the constraints as columns.
        returns : Matrix containing basis of tangent space as its columns.
        """
        Jac=self.Q(x).reshape((self.m+self.d,self.m))
        assert Jac.shape == (self.m + self.d, self.m), "Q must have shape ({}, {}) but found shape {}".format(self.m+self.d, self.m, Jac.shape)
        return svd(Jac)[0][:, self.m:]

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

