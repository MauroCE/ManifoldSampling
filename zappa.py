import numpy as np
from numpy.random import randn, rand
from numpy.linalg import svd
from numpy import log, zeros
import matplotlib.pyplot as plt
from numpy import pi
from scipy.optimize import root
from scipy.stats import multivariate_normal, norm
from utils import normalize

def zappa_sampling(x0, manifold, logf, logp, n, sigma, tol, a_guess):
    """
    Samples from a manifold using the Zappa algorithm. 

    x0 : Numpy Array 
         Initial point on the manifold. Has dimension (d,) where d is the manifold dimension.

    manifold : Manifold
               Object of Manifold class.

    logf : callable
           Function computing log-target density constrained on the manifold. In most cases, this 
           is uniform and the function should output log(1) = 0. Takes as input a vector of 
           dimension (d + m, )

    logp : callable
           Function computing log-proposal density. In most cases this is a d-dimensional 
           isotropic Gaussian. Takes as input a vector of dimensions (d + m, ).

    n : Int
        Number of samples

    sigma : Float
            Scale for proposal.

    tol : Float
          Tolerance for root-finding algorithm used to find a.

    a_guess : Float
              Initial guess for a. Used by the root-finding algorithm.

    Returns the samples as an array of dimensions (n, d + m)
    """
    # Check arguments
    n = int(n)
    d, m = manifold.get_dimension(), manifold.get_codimension()

    # Initial point on the manifold
    x = x0

    # House-keeping
    samples = zeros((n, d + m))    # Store n samples on the manifold
    samples[0, :] = x
    i = 1

    # Log-uniforms for MH accept-reject step
    logu = log(rand(n))

    # Run until you get n samples
    while i < n:

        # Compute gradient, gradient basis & tangent basis at x
        Qx = manifold.Q(x)                       # Gradient at x.                             Size: (d + m, )
        gx_basis = normalize(Qx)                 # ON basis for gradient at x                 Size: (d + m, )
        tx_basis = manifold.tangent_basis(Qx) # ON basis for tangent space at x using SVD  Size: (d + m, d)

        # Sample along tangent 
        v_sample = sigma*randn(d)  # Isotropic MVN with scaling sigma         Size: (d, )
        v = tx_basis @ v_sample    # Linear combination of the basis vectors  Size: (d + m, )

        # Forward Projection
        a, flag = project(x, v, Qx, manifold.q, tol, a_guess)
        if flag == 0:                                  # Projection failed
            samples[i, :] = x                          # Stay at x
            i += 1
            continue
        y = x + v + a*Qx.flatten()                              # Projected point (d + m, )

        # Compute v' from y
        Qy = manifold.Q(y)                         # Gradient at y.                            Size: (d + m, )
        gy_basis = normalize(Qy)             # ON basis for gradient at y                Size: (d + m, )
        ty_basis = manifold.tangent_basis(Qy)      # ON basis for tangent space at y using SVD Size: (d + m, d)
        v_prime_sample = (x - y) @ ty_basis  # Components along tangent                  Size: (d + m, )

        # Metropolis-Hastings
        if logu[i] > logf(y) + logp(v_prime_sample) - logf(x) - logp(v_sample):
            samples[i, :] = x     # Reject. Stay at x
            i += 1
            continue

        # Backward Projection
        v_prime = v_prime_sample @ ty_basis.T   # Linear combination of the basis vectors. Size: (d + m, )
        a_prime, flag = project(y, v_prime, Qy, manifold.q, tol, a_guess)
        if flag == 0:
            samples[i, :] = x     # projection failed, stay at x
            i += 1
            continue

        # Accept move
        x = y
        samples[i, :] = x
        i += 1

    return samples

def project(x, v, Q, q, tol=None, a_guess=1):
    """Finds a such that q(x + v + a*Q) = 0"""
    if len(Q.shape) == 2:
        Q = Q.flatten()
    opt_output = root(lambda a: q(x + v + a*Q), a_guess, tol=tol)
    return (opt_output.x, opt_output.success) # output flag for accept/reject