import numpy as np
from numpy.random import randn, rand
from numpy.linalg import svd
import numpy.linalg as la
from numpy import log, zeros
import matplotlib.pyplot as plt
from numpy import pi
from scipy.optimize import root
from scipy.stats import multivariate_normal, norm
from utils import normalize
from scipy.optimize.minpack import _root_hybr


def zappa_sampling(x0, manifold, logf, logp, n, sigma, tol, a_guess, root=None, maxiter=50):
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
    # Choose which function to use to project onto the manifold
    project = project_original

    if root == 'root':
        project = lambda x, v, Q, q, tol, a_guess, maxiter: project_root(x, v, Q, q, manifold.Q, tol, a_guess, maxiter)
    elif root == 'newton':
        project = lambda x, v, Q, q, tol, a_guess, maxiter: project_newton(x, v, Q, q, manifold.Q, tol, a_guess, maxiter)

    # Check arguments
    n = int(n)
    d, m = manifold.get_dimension(), manifold.get_codimension()

    # Initial point on the manifold
    x = x0

    # housekeeping
    samples = zeros((n, d + m))    # Store n samples on the manifold
    samples[0, :] = x
    i = 1
    # Store number of function and jacobian evaluations for x (and statuses)
    n_fun_eval_x = []
    n_jac_eval_x = []
    statuses_x = []
    # Same, but for y
    n_fun_eval_y = []
    n_jac_eval_y = []
    statuses_y = []

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
        a, flag, nfev, njev, status = project(x, v, Qx, manifold.q, tol, a_guess, maxiter)
        # Housekeeping
        n_fun_eval_x.append(nfev)
        n_jac_eval_x.append(njev)
        statuses_x.append(status)
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
        a_prime, flag, nfev, njev, status = project(y, v_prime, Qy, manifold.q, tol, a_guess, maxiter)
        # Housekeeping
        n_fun_eval_y.append(nfev)
        n_jac_eval_y.append(njev)
        statuses_y.append(status)
        if flag == 0:
            samples[i, :] = x     # projection failed, stay at x
            i += 1
            continue

        # Accept move
        x = y
        samples[i, :] = x
        i += 1
    output = {
        'samples': samples,
        'nfevx': n_fun_eval_x,
        'njevx': n_jac_eval_x,
        'nfevy': n_fun_eval_y,
        'njevy': n_jac_eval_y,
        'xstatus': statuses_x,
        'ystatus': statuses_y
    }
    return output


def project_original(x, v, Q, q, tol=None, a_guess=1, maxiter=50):
    """Finds a such that q(x + v + a*Q) = 0"""
    if len(Q.shape) == 2:
        Q = Q.flatten()
    opt_output = root(lambda a: q(x + v + a*Q), a_guess, tol=tol, options={'maxfev':maxiter})
    return (opt_output.x, opt_output.success, 0, 0, 0) # output flag for accept/reject


def project_root(x, v, Q, q, grad_q, tol=None, a_guess=1, maxiter=50):
    """Finds a such that q(x + v + a*Q) = 0"""
    #if len(Q.shape) == 2:
    #    Q = Q.flatten()
    ######obj = lambda a: q(x + v + Q @ a)
    ######jac = lambda a: grad_q(x + v + Q @ a).T @ Q
    ######a0  = np.array([a_guess])
    #opt_output = root(obj, a0, jac=jac, tol=tol, options={'maxfev':maxiter, 'col_deriv': True})
    # INFO: https://www.math.utah.edu/software/minpack/minpack/hybrj.html
    # Inputs: Objecti Function, Jacobian Function, Initial Guess, additional args, full output?, col_deriv?, tol, maxiter
    #sol, out_dict, status = _hybrj(obj, jac, a0, (), 1, 0, tol, maxiter)
    #return sol, status==1, out_dict['nfev'], out_dict['njev'], status  # Status == 0 means Success
    ##sol, out_dict, status = _hybrd(obj, a0, (), 1, tol, maxiter)
    out = _root_hybr(lambda a: q(x + v + Q @ a), a_guess, jac=lambda a: grad_q(x + v + Q @ a).T @ Q, col_deriv=True, xtol=tol, maxfev=maxiter)
    #out = _root_hybr(obj, a0, tol=tol, options={'maxfev':maxiter, 'col_deriv': True})
    # Results to output
    #return (sol, status, out_dict['nfev'], out_dict['njev']) # Solution, Success==True, Number Obj Func evals, Number Jacobian evals
    ##return (sol, status, out_dict['nfev'], 0)
    return out.x, out.success, out.nfev, out.njev, out.status

def project_newton(x, v, Q, q, grad_q, tol=None, a_guess=0.0, maxiter=50):
    """Finds a such that q(x + v + a*Q) = 0 using Newton's method"""
    # Housekeeping
    a = zeros(Q.shape[1])   # (m, ) One entry for each constraint
    flag = 1
    i = 0
    # Newton's algorithm
    while la.norm(q(x + v + Q @ a)) > tol:
        # Notice that I need to wrap -q(x + v + Q @ a) into a Numpy Array because it returns a scalar.
        # Otherwise la.solve() will complain. This will NOT work for m > 1 constraints. In that case,
        # I would need to rewrite the function q of the manifold to return an array of size (m,)
        delta_a = la.solve(grad_q(x + v + Q @ a).T @ Q, np.array([-q(x + v + Q @ a)]))
        a += delta_a
        i += 1
        if i > maxiter:
            flag = 0
            return a, flag, i, i, 0
    return a, flag, i, i, 0
