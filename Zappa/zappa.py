"""
Simple, Efficient Zappa sampling. Returns samples only.
"""
import numpy as np
from numpy.random import randn, rand, exponential
from numpy.linalg import svd, solve
from numpy import log, zeros
import matplotlib.pyplot as plt
from numpy import pi
from scipy.optimize import root
from scipy.stats import multivariate_normal, norm
from utils import normalize
from utils import logp as logp_scale
from utils import angle_between
import scipy.linalg as la

def zappa_sampling_multivariate(x0, manifold, logf, logp, n, sigma, tol, a_guess, maxiter=50):
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
              This should have dimension (m, )

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

    # Do First Step
    Qx = manifold.Q(x)                       # Gradient at x.                             Size: (d + m, )
    tx_basis = manifold.tangent_basis(Qx)    # ON basis for tangent space at x using SVD  Size: (d + m, d)

    # Run until you get n samples
    while i < n:

        # Sample along tangent 
        v_sample = sigma*randn(d)  # Isotropic MVN with scaling sigma         Size: (d, )
        v = tx_basis @ v_sample    # Linear combination of the basis vectors  Size: (d + m, )

        # Forward Projection
        a, flag = project_multivariate(x, v, Qx, manifold.q, tol, a_guess, maxiter)
        if flag == 0:                                  # Projection failed
            samples[i, :] = x                          # Stay at x
            i += 1
            continue
        y = x + v + Qx @ a                             # Projected point (d + m, )

        # Compute v' from y
        Qy = manifold.Q(y)                         # Gradient at y.                            Size: (d + m, )
        ty_basis = manifold.tangent_basis(Qy)      # ON basis for tangent space at y using SVD Size: (d + m, d)
        v_prime_sample = (x - y) @ ty_basis  # Components along tangent                  Size: (d + m, )

        # Metropolis-Hastings
        if logu[i] > logf(y) + logp(v_prime_sample) - logf(x) - logp(v_sample):
            samples[i, :] = x     # Reject. Stay at x
            i += 1
            continue

        # Backward Projection
        v_prime = v_prime_sample @ ty_basis.T   # Linear combination of the basis vectors. Size: (d + m, )
        a_prime, flag = project_multivariate(y, v_prime, Qy, manifold.q, tol, a_guess, maxiter)
        if flag == 0:
            samples[i, :] = x     # projection failed, stay at x
            i += 1
            continue

        # Accept move x --> y
        x = y
        samples[i, :] = x
        Qx = Qy                # Store gradient
        tx_basis = ty_basis    # Store tangent basis
        i += 1

    return samples



def zappa_step_EJSD_deterministic(x0, v0, logu, manifold, logf, logp, sigma, tol, a_guess, maxiter=50):
    """
    One step of Zappa, deterministic, computing EJSD.
    """
    
    a = 0.0
    # Do First Step
    Qx = manifold.Q(x0)                       # Gradient at x.                             Size: (d + m, )
    tx_basis = manifold.tangent_basis(Qx)    # ON basis for tangent space at x using SVD  Size: (d + m, d)
    total_n_gradients = 1.0

    # Sample along tangent 
    v_sample = sigma*v0  # Isotropic MVN with scaling sigma         Size: (d, )
    v = tx_basis @ v_sample    # Linear combination of the basis vectors  Size: (d + m, )

    # Forward Projection
    a, flag, n_grad = project_zappa(manifold.q, x0 + v, Qx, manifold.Q, tol, maxiter)
    total_n_gradients += n_grad
    if flag == 0:                                  # Projection failed
        return x0, 0.0, 0.0, total_n_gradients
    y = x0 + v + Qx @ a                             # Projected point (d + m, )

    # Compute v' from y
    Qy = manifold.Q(y)                         # Gradient at y.                            Size: (d + m, )
    ty_basis = manifold.tangent_basis(Qy)      # ON basis for tangent space at y using SVD Size: (d + m, d)
    v_prime_sample = (x0 - y) @ ty_basis  # Components along tangent                  Size: (d + m, )
    total_n_gradients += 1

    # Metropolis-Hastings
    loga = logf(y) + logp(v_prime_sample) - logf(x0) - logp(v_sample)
    a = min(1.0, np.exp(loga))
    EJSD = a * np.linalg.norm(x0 - y)**2
    if logu > loga:
        return x0, a, EJSD, total_n_gradients

    # Backward Projection
    v_prime = v_prime_sample @ ty_basis.T   # Linear combination of the basis vectors. Size: (d + m, )
    a_prime, flag, n_grad = project_zappa(manifold.q, y + v_prime, Qy, manifold.Q, tol, maxiter)
    total_n_gradients += n_grad
    if flag == 0:
        return x0, a, EJSD, total_n_gradients

    return y, a, EJSD, total_n_gradients


def zappa_step_accept(x0, manifold, p, n, tol, a_guess, maxiter=50):
    """
    Samples using Zappa.
    """
    # Check arguments
    n = int(n)
    d, m = manifold.get_dimension(), manifold.get_codimension()
    logf = lambda xy: - log(np.linalg.norm(manifold.Q(xy)))

    # Initial point on the manifold
    x = x0

    # House-keeping
    samples = zeros((n, d + m))    # Store n samples on the manifold
    accept = zeros(n)
    i = 0

    # Log-uniforms for MH accept-reject step
    logu = log(rand(n))

    # Do First Step
    Qx = manifold.Q(x)                       # Gradient at x.                             Size: (d + m, )
    tx_basis = manifold.tangent_basis(Qx)    # ON basis for tangent space at x using SVD  Size: (d + m, d)

    # Run until you get n samples
    while i < n:

        # Sample along tangent 
        v_sample = p.rvs().reshape([-1])        # Isotropic MVN with scaling sigma         Size: (d, )
        v = tx_basis @ v_sample    # Linear combination of the basis vectors  Size: (d + m, )

        # Forward Projection
        a, flag = project_multivariate(x, v, Qx, manifold.q, tol, a_guess, maxiter)
        if flag == 0:                                  # Projection failed
            samples[i, :] = x                          # Stay at x
            accept[i] = 0
            i += 1
            continue
        y = x + v + Qx @ a                             # Projected point (d + m, )

        # Compute v' from y
        Qy = manifold.Q(y)                         # Gradient at y.                            Size: (d + m, )
        ty_basis = manifold.tangent_basis(Qy)      # ON basis for tangent space at y using SVD Size: (d + m, d)
        v_prime_sample = (x - y) @ ty_basis  # Components along tangent                  Size: (d + m, )

        # Metropolis-Hastings
        if logu[i] > logf(y) + p.logpdf(v_prime_sample) - logf(x) - p.logpdf(v_sample):
            samples[i, :] = x     # Reject. Stay at x
            accept[i] = 0
            i += 1
            continue

        # Backward Projection
        v_prime = v_prime_sample @ ty_basis.T   # Linear combination of the basis vectors. Size: (d + m, )
        _, flag = project_multivariate(y, v_prime, Qy, manifold.q, tol, a_guess, maxiter)
        if flag == 0:
            samples[i, :] = x     # projection failed, stay at x
            accept[i] = 0
            i += 1
            continue

        # Accept move x --> y
        accept[i] = 1
        x = y
        samples[i, :] = x
        Qx = Qy                # Store gradient
        tx_basis = ty_basis    # Store tangent basis
        i += 1

    return samples, accept


def zappa_step_acceptPC(x0, manifold, p, A, n, tol, a_guess, maxiter=50):
    """
    Samples using Zappa.
    """
    # Check arguments
    n = int(n)
    d, m = manifold.get_dimension(), manifold.get_codimension()
    logf = lambda xy: - log(np.linalg.norm(A @ manifold.Q(A.T @ xy)))

    # Initial point on the manifold
    x = solve(A.T, x0)

    # New constraint
    z_tilde = multivariate_normal(np.zeros(d+m), np.eye(d+m)).pdf(x)
    q_tilde = lambda xy: np.linalg.norm(xy)**2 + (d+m)*log(2*np.pi) + 2*log(z_tilde)

    # House-keeping
    samples = zeros((n, d + m))    # Store n samples on the manifold
    accept = zeros(n)
    i = 0

    # Log-uniforms for MH accept-reject step
    logu = log(rand(n))

    # Do First Step
    Qx = A @ manifold.Q(x)                       # Gradient at x.                             Size: (d + m, )
    tx_basis = manifold.tangent_basis(Qx)    # ON basis for tangent space at x using SVD  Size: (d + m, d)

    # Run until you get n samples
    while i < n:

        # Sample along tangent 
        v_sample = p.rvs().reshape([-1])  # Isotropic MVN with scaling sigma         Size: (d, )
        v = tx_basis @ v_sample           # Linear combination of the basis vectors  Size: (d + m, )

        
        # Forward Projection
        a, flag = project_multivariate(x, v, Qx, q_tilde, tol, a_guess, maxiter)
        if flag == 0:                                  # Projection failed
            samples[i, :] = x                          # Stay at x
            accept[i] = 0
            i += 1
            continue
        y = x + v + Qx @ a                             # Projected point (d + m, )

        # Compute v' from y
        Qy = A @ manifold.Q(y)                     # Gradient at y.                            Size: (d + m, )
        ty_basis = manifold.tangent_basis(Qy)      # ON basis for tangent space at y using SVD Size: (d + m, d)
        v_prime_sample = (x - y) @ ty_basis  # Components along tangent                  Size: (d + m, )

        # Metropolis-Hastings
        if logu[i] > logf(y) + p.logpdf(v_prime_sample) - logf(x) - p.logpdf(v_sample):
            samples[i, :] = x     # Reject. Stay at x
            accept[i] = 0
            i += 1
            continue

        # Backward Projection
        v_prime = v_prime_sample @ ty_basis.T   # Linear combination of the basis vectors. Size: (d + m, )
        _, flag = project_multivariate(y, v_prime, Qy, q_tilde, tol, a_guess, maxiter)
        if flag == 0:
            samples[i, :] = x     # projection failed, stay at x
            accept[i] = 0
            i += 1
            continue

        # Accept move x --> y
        accept[i] = 1
        x = y
        samples[i, :] = x
        Qx = Qy                # Store gradient
        tx_basis = ty_basis    # Store tangent basis
        i += 1

    # At the end, we must transform them back
    samples = samples @ A   # samples @ (A.T).T as it would be A.T @ xy
    return samples, accept


def project_multivariate(x, v, Q, q, tol=None, a_guess=np.zeros(1), maxiter=50):
    """Finds a such that q(x + v + a*Q) = 0"""
    opt_output = root(lambda a: q(x + v + Q @ a), a_guess, tol=tol, options={'maxfev':maxiter})
    return (opt_output.x, opt_output.success) # output flag for accept/reject


def project_zappa(q, z, Q, grad_q, tol = 1.48e-08 , maxiter = 50):
    '''
    This version is the version of Miranda & Zappa. It retuns i, the number of iterations
    i.e. the number of gradient evaluations used.
    '''
    a, flag, i = np.zeros(Q.shape[1]), 1, 0

    #Newton's method to solve q(z + Q @ a)
    while la.norm(q(z + Q @ a)) > tol:
        delta_a = la.solve(grad_q(z + Q @ a).transpose() @ Q, -q(z + Q @ a))
        a += delta_a
        i += 1
        #print(a, q(z + Q @ a), i) #for debugging
        if i > maxiter: 
            flag = 0
            return a, flag, i
            
    return a, flag, i





def zappa_sampling(x0, manifold, logf, logp, n, sigma, tol, a_guess, maxiter=50):
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

    # Do First Step
    Qx = manifold.Q(x)                       # Gradient at x.                             Size: (d + m, )
    tx_basis = manifold.tangent_basis(Qx)    # ON basis for tangent space at x using SVD  Size: (d + m, d)

    # Run until you get n samples
    while i < n:

        # Sample along tangent 
        v_sample = sigma*randn(d)  # Isotropic MVN with scaling sigma         Size: (d, )
        v = tx_basis @ v_sample    # Linear combination of the basis vectors  Size: (d + m, )

        # Forward Projection
        a, flag = project(x, v, Qx, manifold.q, tol, a_guess, maxiter)
        if flag == 0:                                  # Projection failed
            samples[i, :] = x                          # Stay at x
            i += 1
            continue
        y = x + v + a*Qx.flatten()                              # Projected point (d + m, )

        # Compute v' from y
        Qy = manifold.Q(y)                         # Gradient at y.                            Size: (d + m, )
        ty_basis = manifold.tangent_basis(Qy)      # ON basis for tangent space at y using SVD Size: (d + m, d)
        v_prime_sample = (x - y) @ ty_basis  # Components along tangent                  Size: (d + m, )

        # Metropolis-Hastings
        if logu[i] > logf(y) + logp(v_prime_sample) - logf(x) - logp(v_sample):
            samples[i, :] = x     # Reject. Stay at x
            i += 1
            continue

        # Backward Projection
        v_prime = v_prime_sample @ ty_basis.T   # Linear combination of the basis vectors. Size: (d + m, )
        a_prime, flag = project(y, v_prime, Qy, manifold.q, tol, a_guess, maxiter)
        if flag == 0:
            samples[i, :] = x     # projection failed, stay at x
            i += 1
            continue

        # Accept move x --> y
        x = y
        samples[i, :] = x
        Qx = Qy                # Store gradient
        tx_basis = ty_basis    # Store tangent basis
        i += 1

    return samples

def project(x, v, Q, q, tol=None, a_guess=1, maxiter=50):
    """Finds a such that q(x + v + a*Q) = 0"""
    opt_output = root(lambda a: q(x + v + Q @ a), np.array([a_guess]), tol=tol, options={'maxfev':maxiter})
    return (opt_output.x, opt_output.success) # output flag for accept/reject

def zappa_adaptive(x0, manifold, logf, n, s0, tol, a_guess, ap_star, update_scale, maxiter=50):
    """
    Adaptive version of Zappa Sampling. It uses Stochastic Approximation. It does not use Polyak averaging. 

    s0 : float
         Initial scale
    ap_star : float
              Target acceptance probability. Should be between 0 and 1.
    update_scale : Callable
                   Function that updates the scale.
    """
    # Check arguments
    n = int(n)
    d, m = manifold.get_dimension(), manifold.get_codimension()

    # Initial point on the manifold & initial scale for tangent sampling
    x = x0
    s = s0   
    l = np.log(s0)

    # House-keeping
    samples = zeros((n, d + m))    # Store n samples on the manifold
    samples[0, :] = x
    i = 1

    # Log-uniforms for MH accept-reject step
    logu = log(rand(n))

    # Do first step
    Qx = manifold.Q(x)                       # Gradient at x.                             Size: (d + m, )
    gx_basis = normalize(Qx)                 # ON basis for gradient at x                 Size: (d + m, )
    tx_basis = manifold.tangent_basis(Qx)    # ON basis for tangent space at x using SVD  Size: (d + m, d)

    # Define proposal distribution
    logp = lambda xy: logp_scale(xy, s)

    while i < n:
        # Initiate log acceptance ratio to be -inf (i.e. alpha(x, u) = 0)
        log_ap = -np.inf

        # Sample along tangent 
        v_sample = s*randn(d)  # Isotropic MVN with scaling sigma         Size: (d, )
        v = tx_basis @ v_sample    # Linear combination of the basis vectors  Size: (d + m, )

        # Forward Projection
        a, flag = project(x, v, Qx, manifold.q, tol, a_guess, maxiter)
        if flag == 0:                                              # Projection failed
            samples[i, :] = x                                      # Stay at x
            s, l = update_scale(np.exp(log_ap), ap_star, i + 1, l) # Update log scale and scale
            i += 1
            continue
        y = x + v + a*Qx.flatten()                              # Projected point (d + m, )

        # Compute v' from y
        Qy = manifold.Q(y)                         # Gradient at y.                            Size: (d + m, )
        gy_basis = normalize(Qy)             # ON basis for gradient at y                Size: (d + m, )
        ty_basis = manifold.tangent_basis(Qy)      # ON basis for tangent space at y using SVD Size: (d + m, d)
        v_prime_sample = (x - y) @ ty_basis  # Components along tangent                  Size: (d + m, )

        # Metropolis-Hastings 
        log_ap = logf(y) + logp(v_prime_sample) - logf(x) - logp(v_sample)
        if logu[i] > log_ap:  #logf(y) + logp(v_prime_sample) - logf(x) - logp(v_sample):
            samples[i, :] = x     # Reject. Stay at x
            s, l = update_scale(np.exp(log_ap), ap_star, i + 1, l) # Update log scale and scale
            i += 1
            continue

        # Backward Projection
        v_prime = v_prime_sample @ ty_basis.T   # Linear combination of the basis vectors. Size: (d + m, )
        a_prime, flag = project(y, v_prime, Qy, manifold.q, tol, a_guess, maxiter)
        if flag == 0:
            samples[i, :] = x     # projection failed, stay at x
            s, l = update_scale(np.exp(log_ap), ap_star, i + 1, l)   # Update log scale and scale
            i += 1
            continue

        # Accept move
        x = y
        samples[i, :] = x
        s, l = update_scale(np.exp(log_ap), ap_star, i + 1, l)
        Qx = Qy
        gx_basis = gy_basis
        tx_basis = ty_basis
        i += 1

    return samples

def zappa_adaptive_returnscale(x0, manifold, logf, n, s0, tol, a_guess, ap_star, update_scale, maxiter=50):
    """
    Adaptive version of Zappa Sampling. It uses Stochastic Approximation. It does not use Polyak averaging. 
    THIS VERSION ALSO OUTPUTS THE FINAL SCALE.

    s0 : float
         Initial scale
    ap_star : float
              Target acceptance probability. Should be between 0 and 1.
    update_scale : Callable
                   Function that updates the scale.
    """
    # Check arguments
    n = int(n)
    d, m = manifold.get_dimension(), manifold.get_codimension()

    # Initial point on the manifold & initial scale for tangent sampling
    x = x0
    s = s0   
    l = np.log(s0)

    # House-keeping
    samples = zeros((n, d + m))    # Store n samples on the manifold
    samples[0, :] = x
    i = 1

    # Log-uniforms for MH accept-reject step
    logu = log(rand(n))

    # Do First Step
    Qx = manifold.Q(x)                       # Gradient at x.                             Size: (d + m, )
    gx_basis = normalize(Qx)                 # ON basis for gradient at x                 Size: (d + m, )
    tx_basis = manifold.tangent_basis(Qx)    # ON basis for tangent space at x using SVD  Size: (d + m, d)    

    # Define proposal distribution
    logp = lambda xy: logp_scale(xy, s)

    while i < n:
        # Initiate log acceptance ratio to be -inf (i.e. alpha(x, u) = 0)
        log_ap = -np.inf

        # Sample along tangent 
        v_sample = s*randn(d)  # Isotropic MVN with scaling sigma         Size: (d, )
        v = tx_basis @ v_sample    # Linear combination of the basis vectors  Size: (d + m, )

        # Forward Projection
        a, flag = project(x, v, Qx, manifold.q, tol, a_guess, maxiter)
        if flag == 0:                                              # Projection failed
            samples[i, :] = x                                      # Stay at x
            s, l = update_scale(np.exp(log_ap), ap_star, i + 1, l) # Update log scale and scale
            i += 1
            continue
        y = x + v + a*Qx.flatten()                              # Projected point (d + m, )

        # Compute v' from y
        Qy = manifold.Q(y)                         # Gradient at y.                            Size: (d + m, )
        gy_basis = normalize(Qy)             # ON basis for gradient at y                Size: (d + m, )
        ty_basis = manifold.tangent_basis(Qy)      # ON basis for tangent space at y using SVD Size: (d + m, d)
        v_prime_sample = (x - y) @ ty_basis  # Components along tangent                  Size: (d + m, )

        # Metropolis-Hastings 
        log_ap = logf(y) + logp(v_prime_sample) - logf(x) - logp(v_sample)
        if logu[i] > logf(y) + logp(v_prime_sample) - logf(x) - logp(v_sample):
            samples[i, :] = x     # Reject. Stay at x
            s, l = update_scale(np.exp(log_ap), ap_star, i + 1, l) # Update log scale and scale
            i += 1
            continue

        # Backward Projection
        v_prime = v_prime_sample @ ty_basis.T   # Linear combination of the basis vectors. Size: (d + m, )
        a_prime, flag = project(y, v_prime, Qy, manifold.q, tol, a_guess, maxiter)
        if flag == 0:
            samples[i, :] = x     # projection failed, stay at x
            s, l = update_scale(np.exp(log_ap), ap_star, i + 1, l)   # Update log scale and scale
            i += 1
            continue

        # Accept move
        x = y
        samples[i, :] = x
        s, l = update_scale(np.exp(log_ap), ap_star, i + 1, l)
        Qx = Qy 
        gx_basis = gy_basis
        tx_basis = ty_basis 
        i += 1

    return samples, s

def zappa_persistent(x0, manifold, logf, logp, n, sigma, tol, a_guess, maxiter=50):
    """
    Samples from a manifold using the Zappa algorithm but this time it is PERSISTENT.
    This means that we go in one direction. To achieve this, we sample from an exponential 
    distribution. 

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

    # Do the first step
    Qx = manifold.Q(x)                      # Gradient at x.                             Size: (d + m, )
    gx_basis = normalize(Qx)                # ON basis for gradient at x                 Size: (d + m, )
    tx_basis = manifold.tangent_basis(Qx)   # ON basis for tangent space at x using SVD  Size: (d + m, d)

    # Run until you get n samples
    while i < n:

        # Sample along tangent 
        v_sample = sigma*exponential(size=d)        # To get persistece, we don't sample.      Size: (d, )
        # I need to compute both tries and see which one works (?)

        v = tx_basis @ v_sample                # Linear combination of the basis vectors  Size: (d + m, )

        # Forward Projection
        a, flag = project(x, v, Qx, manifold.q, tol, a_guess, maxiter)
        if flag == 0:    
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
        if logu[i] > logf(y) + logp(abs(v_prime_sample)) - logf(x) - logp(v_sample):
            samples[i, :] = x     # Reject. Stay at x
            i += 1
            continue

        # Backward Projection
        v_prime = v_prime_sample @ ty_basis.T   # Linear combination of the basis vectors. Size: (d + m, )
        a_prime, flag = project(y, v_prime, Qy, manifold.q, tol, a_guess, maxiter)
        if flag == 0:
            samples[i, :] = x     # projection failed, stay at x
            i += 1
            continue

        # Accept move x --> y
        x = y
        samples[i, :] = x
        Qx = Qy
        gx_basis = gy_basis
        # Change the direction of the basis to make sure it is in the same direction as the previous one
        # TODO: This only works in 2D!!
        angle1 = angle_between(tx_basis.flatten(), ty_basis.flatten())
        angle2 = angle_between(tx_basis.flatten(), -ty_basis.flatten())
        tx_basis = ty_basis * np.sign(angle2 - angle1)    
        i += 1

    return samples

def zappa_projectv(x0, manifold, logf, logp, n, sigma, tol, a_guess, refreshrate=0.1, maxiter=50):
    """
    Samples from a manifold using the Zappa algorithm. This time it projects the velocity onto the tangent plane of where it has landed.

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

    refreshrate: Float
                 At which rate to refresh the velocity. This is needed to avoid the algorithm
                 getting stuck with the same very small velocity. Indeed if the velocity is small
                 then it will always be accepted but it will explore the space very poorly.

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
    # Velocities
    #vs = np.zeros((n, d + m))

    # Log-uniforms for MH accept-reject step
    logu = log(rand(n))

    # Do First Step
    Qx = manifold.Q(x)                       # Gradient at x.                             Size: (d + m, )
    gx_basis = normalize(Qx)                 # ON basis for gradient at x                 Size: (d + m, )
    tx_basis = manifold.tangent_basis(Qx)    # ON basis for tangent space at x using SVD  Size: (d + m, d)

    # Sample along tangent 
    v_sample = sigma*randn(d)  # Isotropic MVN with scaling sigma         Size: (d, )
    v = tx_basis @ v_sample    # Linear combination of the basis vectors  Size: (d + m, )
    #vs[0, :] = v

    # Run until you get n samples
    while i < n:

        # Forward Projection
        a, flag = project(x, v, Qx, manifold.q, tol, a_guess, maxiter)
        if flag == 0:                                  # Projection failed
            samples[i, :] = x                          # Stay at x
            # Sample new velocity at x.
            v_sample = sigma*randn(d)  # Isotropic MVN with scaling sigma         Size: (d, )
            v = tx_basis @ v_sample    # Linear combination of the basis vectors  Size: (d + m, )
            #vs[i, :] = v
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
            # Sample new velocity
            v_sample = sigma*randn(d)  # Isotropic MVN with scaling sigma         Size: (d, )
            v = tx_basis @ v_sample    # Linear combination of the basis vectors  Size: (d + m, )
            #vs[i, :] = v
            i += 1
            continue

        # Backward Projection
        v_prime = v_prime_sample @ ty_basis.T   # Linear combination of the basis vectors. Size: (d + m, )
        a_prime, flag = project(y, v_prime, Qy, manifold.q, tol, a_guess, maxiter)
        if flag == 0:
            samples[i, :] = x     # projection failed, stay at x
            # Sample new velocity
            v_sample = sigma*randn(d)  # Isotropic MVN with scaling sigma         Size: (d, )
            v = tx_basis @ v_sample    # Linear combination of the basis vectors  Size: (d + m, )
            #vs[i, :] = v
            i += 1
            continue

        # Accept move x --> y
        x = y
        samples[i, :] = x
        Qx = Qy                # Store gradient
        gx_basis = gy_basis    # Store gradient basis
        tx_basis = ty_basis    # Store tangent basis
        # If the move has been accepted, simply project the current velocity onto
        # the new tangent plane. TODO: Check if v_prime_sample is just - this.
        # We need to refresh the velocity at a refreshment rate! 
        if np.random.rand() < refreshrate:
            # Refresh!
            v_sample = sigma*randn(d)
            v = tx_basis @ v_sample
        else:
            # Project!
            v_sample = v @ ty_basis
            v = v_sample @ ty_basis.T
        #vs[i, :] = v
        # If it has been accepted, use update a_guess as there's a high chance it might improve things
        a_guess = a 
        i += 1

    return samples  #, vs

def zappa_projectv_adaptive(x0, manifold, logf, n, s0, tol, a_guess, ap_star, update_scale, refreshrate=0.1, maxiter=50):
    """
    Adaptive version of Zappa where we project the velocity onto the tangent plane of where it has landed.

    x0 : Numpy Array 
         Initial point on the manifold. Has dimension (d,) where d is the manifold dimension.

    manifold : Manifold
               Object of Manifold class.

    logf : callable
           Function computing log-target density constrained on the manifold. In most cases, this 
           is uniform and the function should output log(1) = 0. Takes as input a vector of 
           dimension (d + m, )

    n : Int
        Number of samples

    s0 : Float
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
    s = s0
    l = np.log(s0)

    # House-keeping
    samples = zeros((n, d + m))    # Store n samples on the manifold
    samples[0, :] = x
    i = 1

    # Log-uniforms for MH accept-reject step
    logu = log(rand(n))

    # Do First Step
    Qx = manifold.Q(x)                       # Gradient at x.                             Size: (d + m, )
    gx_basis = normalize(Qx)                 # ON basis for gradient at x                 Size: (d + m, )
    tx_basis = manifold.tangent_basis(Qx)    # ON basis for tangent space at x using SVD  Size: (d + m, d)

    # Define proposal distribution
    logp = lambda xy: logp_scale(xy, s)

    # Sample along tangent 
    v_sample = s*randn(d)      # Isotropic MVN with scaling sigma         Size: (d, )
    v = tx_basis @ v_sample    # Linear combination of the basis vectors  Size: (d + m, )

    # Run until you get n samples
    while i < n:

        # Initiate log acceptance ratio to be -inf (i.e. alpha(x, u) = 0)
        log_ap = -np.inf

        # Forward Projection
        a, flag = project(x, v, Qx, manifold.q, tol, a_guess, maxiter)
        if flag == 0:                                  # Projection failed
            samples[i, :] = x                          # Stay at x
            s, l = update_scale(np.exp(log_ap), ap_star, i + 1, l) # Update log scale and scale
            i += 1
            # Sample new velocity at x.
            v_sample = s*randn(d)           # Isotropic MVN with scaling sigma         Size: (d, )
            v = tx_basis @ v_sample         # Linear combination of the basis vectors  Size: (d + m, )
            continue
        y = x + v + a*Qx.flatten()                              # Projected point (d + m, )

        # Compute v' from y
        Qy = manifold.Q(y)                         # Gradient at y.                            Size: (d + m, )
        gy_basis = normalize(Qy)             # ON basis for gradient at y                Size: (d + m, )
        ty_basis = manifold.tangent_basis(Qy)      # ON basis for tangent space at y using SVD Size: (d + m, d)
        v_prime_sample = (x - y) @ ty_basis  # Components along tangent                  Size: (d + m, )

        # Metropolis-Hastings
        log_ap = logf(y) + logp(v_prime_sample) - logf(x) - logp(v_sample)
        if logu[i] > log_ap:
            samples[i, :] = x     # Reject. Stay at x
            s, l = update_scale(np.exp(log_ap), ap_star, i + 1, l) # Update log scale and scale
            i += 1
            # Sample new velocity
            v_sample = s*randn(d)  # Isotropic MVN with scaling sigma         Size: (d, )
            v = tx_basis @ v_sample    # Linear combination of the basis vectors  Size: (d + m, )
            continue

        # Backward Projection
        v_prime = v_prime_sample @ ty_basis.T   # Linear combination of the basis vectors. Size: (d + m, )
        a_prime, flag = project(y, v_prime, Qy, manifold.q, tol, a_guess, maxiter)
        if flag == 0:
            samples[i, :] = x     # projection failed, stay at x
            s, l = update_scale(np.exp(log_ap), ap_star, i + 1, l)   # Update log scale and scale
            i += 1
            # Sample new velocity
            v_sample = s*randn(d)  # Isotropic MVN with scaling sigma         Size: (d, )
            v = tx_basis @ v_sample    # Linear combination of the basis vectors  Size: (d + m, )
            continue

        # Accept move x --> y
        x = y
        samples[i, :] = x
        s, l = update_scale(np.exp(log_ap), ap_star, i + 1, l)
        Qx = Qy                # Store gradient
        gx_basis = gy_basis    # Store gradient basis
        tx_basis = ty_basis    # Store tangent basis
        # If the move has been accepted, simply project the current velocity onto
        # the new tangent plane. TODO: Check if v_prime_sample is just - this.
        if np.random.rand() < refreshrate:
            # Refresh!
            v_sample = s*randn(d)
            v = tx_basis @ v_sample
        else:
            # Project!
            v_sample = v @ ty_basis
            v = v_sample @ ty_basis.T
        # If it has been accepted, use update a_guess as there's a high chance it might improve things
        a_guess = a 
        i += 1

    return samples