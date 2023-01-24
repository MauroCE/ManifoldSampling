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
from scipy.linalg import qr


def zappa_sampling_storecomps_rattle_manifold(x0, manifold, n, T, B, tol, rev_tol, maxiter=50, norm_ord=2):
    """C-RWM Rattle. This was used in G and K experiment."""
    assert type(B) == int
    assert norm_ord in [2, np.inf]
    assert len(x0) == manifold.n, "Initial point has wrong dimension."
    # Check arguments
    n = int(n)  
    B = int(B)
    δ = T / B
    d, m = manifold.get_dimension(), manifold.get_codimension()

    # Initial point on the manifold
    x = x0
    compute_J = lambda x: manifold.J(x) #manifold.Q(x).T

    # House-keeping
    samples = zeros((n, d + m))    # Store n samples on the manifold
    samples[0, :] = x
    i = 1
    N_EVALS = {'jacobian': 0, 'density': 0}
    ACCEPTED = zeros(n)
    # Define function to compute density
    def logη(x):
        """Computes log density on Manifold but makes sure everything is behaving nicely."""
        return manifold.logη(x) #manifold.logpost(x)

    # Log-uniforms for MH accept-reject step
    logu = log(rand(n))

    # Compute jacobian & density value
    Jx    = compute_J(x) #manifold.Q(x).T
    logηx = logη(x)
    N_EVALS['jacobian'] += 1
    N_EVALS['density'] += 1

    # Function to project onto tangent space
    def qr_project(v, J):
        """Projects using QR decomposition."""
        Q, _ = qr(J.T, mode='economic')
        return Q @ (Q.T @ v)
    
    def linear_project(v, J):
        """Projects by solving linear system."""
        return J.T @ solve(J@J.T, J@v)
        #return J.T.dot(solve(J.dot(J.T), J.dot(v)))

    # Constrained Step Function
    def constrained_rwm_step(x, v, tol, maxiter, Jx, norm_ord=norm_ord):
        """Used for both forward and backward. See Manifold-Lifting paper."""
        # Project momentum
        v_projected = v - linear_project(v, Jx) 
        # Unconstrained position step
        x_unconstr = x + v_projected
        # Position Projection
        a, flag, n_grad = project_zappa_manifold(manifold, x_unconstr, Jx.T, tol, maxiter, norm_ord=norm_ord)
        y = x_unconstr - Jx.T @ a 
        try:
            Jy = compute_J(y) 
        except ValueError as e:
            print("Jacobian computation at projected point failed. ", e)
            return x, v, Jx, 0, n_grad + 1
        # backward velocity
        v_would_have = y - x
        # Find backward momentum & project it to tangent space at new position
        v_projected_endposition = v_would_have - linear_project(v_would_have, Jy) #qr_project(v_would_have, Jy) #qr_project((y - x) / δ, Jy)
        # Return projected position, projected momentum and flag
        return y, v_projected_endposition, Jy, flag, n_grad + 1
    
    def constrained_leapfrog(x0, v0, J0, B, tol, rev_tol, maxiter, norm_ord=norm_ord):
        """Constrained Leapfrog/RATTLE."""
        successful = True
        n_jacobian_evaluations = 0
        x, v, J = x0, v0, J0
        for _ in range(B):
            xf, vf, Jf, converged_fw, n_fw = constrained_rwm_step(x, v, tol, maxiter, J, norm_ord=norm_ord)
            xr, vr , Jr, converged_bw, n_bw = constrained_rwm_step(xf, -vf, tol, maxiter, Jf, norm_ord=norm_ord)
            n_jacobian_evaluations += (n_fw + n_bw)  # +2 due to the line Jy = manifold.Q(y).T
            if (not converged_fw) or (not converged_bw) or (np.linalg.norm(xr - x, ord=norm_ord) >= rev_tol):
                successful = False
                return x0, v0, J0, successful, n_jacobian_evaluations
            else:
                x = xf
                v = vf
                J = Jf
        return x, v, J, successful, n_jacobian_evaluations

    for i in range(n):
        v = δ*randn(m + d) # Sample in the ambient space.
        xp, vp, Jp, LEAPFROG_SUCCESSFUL, n_jac_evals = constrained_leapfrog(x, v, Jx, B, tol=tol, rev_tol=rev_tol, maxiter=maxiter)
        N_EVALS['jacobian'] += n_jac_evals
        if LEAPFROG_SUCCESSFUL:
            logηp = logη(xp)
            N_EVALS['density'] += 1
            if logu[i] <= logηp - logηx - (vp@vp)/2 + (v@v)/2: 
                # Accept
                ACCEPTED[i - 1] = 1
                x, logηx, Jx = xp, logηp, Jp
                samples[i, :] = xp
            else:
                # Reject
                samples[i, :] = x
                ACCEPTED[i - 1] = 0
        else:
            # Reject
            samples[i, :] = x
            ACCEPTED[i - 1] = 0
    return samples, N_EVALS, ACCEPTED


def project_zappa_manifold(manifold, z, Q, tol = 1.48e-08 , maxiter = 50, atol=1e-16, norm_ord=2):
    '''
    USED IN G AND K. This version is the version of Miranda & Zappa. It retuns i, the number of iterations
    i.e. the number of gradient evaluations used.
    '''
    a, flag, i = np.zeros(Q.shape[1]), 1, 0

    # Compute the constrained at z - Q@a. If it fails due to overflow error, return a rejection altogether.
    try:
        projected_value = manifold.q(z - Q@a)
    except ValueError as e:
        return a, 0, i
    # While loop
    while la.norm(projected_value, ord=norm_ord) >= tol:
        try:
            Jproj = manifold.Q(z - Q@a).T
        except ValueError as e:
            print("Jproj failed. ", e)
            return zeros(Q.shape[1]), 0, i
        # Check that Jproj@Q is invertible. Do this by checking condition number 
        # see https://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
        GramMatrix = Jproj@Q
        if np.linalg.cond(GramMatrix) < 1/np.finfo(z.dtype).eps:
            Δa = la.solve(GramMatrix, projected_value)
            a += Δa
            i += 1
            if i > maxiter:
                return zeros(Q.shape[1]), 0, i
            # If we are not at maxiter iteration, compute new projected value
            try:
                projected_value = manifold.q(z - Q@a)
            except ValueError as e:
                return zeros(Q.shape[1]), 0, i
        else:
            # Fail
            return zeros(Q.shape[1]), 0, i
    
    # At the end, check that the found a is not too small.
    # if la.norm(a, ord=norm_ord) <= atol:
    #     return zeros(Q.shape[1]), 0, i
    # else:
    #     return a, 1, i
    return a, 1, i





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

    # Newton's method to solve q(z + Q @ a)
    try:
        while la.norm(q(z + Q @ a)) > tol:
            delta_a = la.solve(grad_q(z + Q @ a).transpose() @ Q, -q(z + Q @ a))
            a += delta_a
            i += 1
            if i > maxiter: 
                flag = 0
                return a, flag, i
    except ValueError as e:
        # print("z: ", z@z)
        # print("Q: ", Q)
        # print("a: ", a, a@a)
        # print("FAILED")
        a, flag = np.zeros(Q.shape[1]), 0
            
    return a, flag, i






def zappa_sampling_storecomps_rattle(x0, manifold, logf, n, T, B, tol, maxiter=50):
    """
    Same as `zappa_sampling_storecomps` except here we don't just do a single step with step size \sigma. 
    Instead, here we basically use RATTLE but with V(x) = constant, so that all the unconstrained momentum
    updates using \nabla V will not be performed. 
    This version of the algorithm allows us to compare it against THUG. Indeed suppose THUG is run with parameters
    T and B (integration time and number of bounces). Then how do you run Zappa? With sigma=T? With sigma=T/B? 
    Using this function, one can run zappa with T and B, hence leading to a fairer comparison. 

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

    T : Float
        Integration time

    B : Int
        Number of RATTLE steps

    tol : Float
          Tolerance for root-finding algorithm used to find a.

    Returns the samples as an array of dimensions (n, d + m)
    """
    assert type(B) == int
    # Check arguments
    n = int(n)
    B = int(B)
    delta = T / B
    d, m = manifold.get_dimension(), manifold.get_codimension()

    # Choose how to compute Qx @ a based on the dimensionality
    if m == 1:
        compute_Qxa = lambda Qx, a: a*Qx.flatten()
    else:
        compute_Qxa = lambda Qx, a: Qx @ a

    # Initial point on the manifold
    x = x0

    # House-keeping
    samples = zeros((n, d + m))    # Store n samples on the manifold
    samples[0, :] = x
    i = 1
    N_EVALS = {'jacobian': 0, 'density': 0}
    ACCEPTED = zeros(n)

    # Log-uniforms for MH accept-reject step
    logu = log(rand(n))

    # Compute jacobian & density value
    Jx    = manifold.Q(x).T
    logfx = logf(x)
    N_EVALS['jacobian'] += 1
    N_EVALS['density'] += 1

    # Function to project onto tangent space
    def qr_project(v, J):
        """Projects using QR decomposition."""
        Q, _ = qr(J.T, mode='economic')
        return Q @ (Q.T @ v)

    # Constrained Step Function
    def constrained_rwm_step(x, v, delta, tol, maxiter, Jx):
        """Used for both forward and backward. See Manifold-Lifting paper."""
        # Project momentum
        v = qr_project(v, Jx)
        # Unconstrained position step
        x = x + delta*v
        # Position Projection
        a, flag, n_grad = project_zappa(manifold.q, x, Jx.T, manifold.Q, tol, maxiter)
        y = x + compute_Qxa(Jx.T, a)
        Jy = manifold.Q(y).T
        # Find backward momentum & project it to tangent space at new position
        v = qr_project((y - x) / delta, Jy)
        # Return projected position, projected momentum and flag
        return y, v, Jy, flag, n_grad


    # Run until you get n samples
    while i < n:
        # Sample velocity in the ambient space & set initial position and velocity
        v_init = randn(m + d)
        x_init = x
        J_init = Jx
        v_init_proj = qr_project(v_init, J_init)
        logfx_init = logfx

        # Set xb and vb
        xb, vb, Jb = x_init, v_init, J_init

        # Basically in this version I want to do B steps of step size delta.
        LEAPFROG_SUCCESSFUL = True
        for _ in range(B):
            x_bp1, v_bp1, J_bp1, flag_fw, n_fw = constrained_rwm_step(xb, vb, delta, tol, maxiter, Jb)
            x_bp1_rv, _, _, flag_rv, n_rv = constrained_rwm_step(x_bp1, v_bp1, -delta, tol, maxiter, J_bp1)
            N_EVALS['jacobian'] += (n_fw + n_rv)
            if flag_fw == 0 or flag_rv == 0 or np.linalg.norm(x_bp1_rv - xb, ord=np.inf) >= tol:
                # Failed, set flag to False so we don't do a MH step and reject immediately
                LEAPFROG_SUCCESSFUL = False
                break
            else: 
                # b step successful. Cache values and continue
                xb = x_bp1
                vb = v_bp1 
                Jb = J_bp1

        if LEAPFROG_SUCCESSFUL:
            # Compute density value
            logf_bp1 = logf(x_bp1)
            # Metropolis-Hastings step
            if logu[i] <= logf_bp1 - logfx_init - (v_bp1@v_bp1)/2 + (v_init_proj@v_init_proj)/2:
                # Accepted, update values
                ACCEPTED[i - 1] = 1
                x     = x_bp1      # Old position      -> New position
                logfx = logf_bp1   # Old density value -> New density value
                Jx    = J_bp1      # Old Jacobian      -> New Jacobian
                # Store sample
                samples[i, :] = x_bp1
                # Update counter
                i += 1
                continue
        else:
            # Rejected, keep same values
            ACCEPTED[i - 1] = 0  # Redundant but helpful for readability
            i += 1
            continue
    return samples, N_EVALS, ACCEPTED




def zappa_sampling_storecomps(x0, manifold, logf, logp, n, sigma, tol, maxiter=50):
    """
    Same as zappa_sampling but we store the number of jacobian computations etc.

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

    Returns the samples as an array of dimensions (n, d + m)
    """
    # Check arguments
    n = int(n)
    d, m = manifold.get_dimension(), manifold.get_codimension()

    # Choose how to compute Qx @ a based on the dimensionality
    if m == 1:
        compute_Qxa = lambda Qx, a: a*Qx.flatten()
    else:
        compute_Qxa = lambda Qx, a: Qx @ a

    # Initial point on the manifold
    x = x0

    # House-keeping
    samples = zeros((n, d + m))    # Store n samples on the manifold
    samples[0, :] = x
    i = 1
    N_EVALS = {'jacobian': 0, 'density': 0}
    ACCEPTED = zeros(n)

    # Log-uniforms for MH accept-reject step
    logu = log(rand(n))

    # Do First Step
    Qx = manifold.Q(x)                       # Gradient at x.                             Size: (d + m, )
    tx_basis = manifold.tangent_basis(Qx)    # ON basis for tangent space at x using SVD  Size: (d + m, d)
    N_EVALS['jacobian'] += 1

    # Compute density now
    logfx = logf(x)
    N_EVALS['density'] += 1

    # Run until you get n samples
    while i < n:

        # Sample along tangent 
        v_sample = sigma*randn(d)  # Isotropic MVN with scaling sigma         Size: (d, )
        v = tx_basis @ v_sample    # Linear combination of the basis vectors  Size: (d + m, )

        # Forward Projection
        a, flag, n_grad = project_zappa(manifold.q, x + v, Qx, manifold.Q, tol, maxiter)
        N_EVALS['jacobian'] += n_grad
        if flag == 0:                                  # Projection failed
            samples[i, :] = x                          # Stay at x
            i += 1
            continue
        y = x + v + compute_Qxa(Qx, a) #a*Qx.flatten()                              # Projected point (d + m, )

        # Compute v' from y
        Qy = manifold.Q(y)                         # Gradient at y.                            Size: (d + m, )
        ty_basis = manifold.tangent_basis(Qy)      # ON basis for tangent space at y using SVD Size: (d + m, d)
        v_prime_sample = (x - y) @ ty_basis  # Components along tangent                  Size: (d + m, )
        N_EVALS['jacobian'] += 1

        # Metropolis-Hastings
        logfy = logf(y)
        N_EVALS['density'] += 1
        if logu[i] > logfy + logp(v_prime_sample) - logfx - logp(v_sample):
            samples[i, :] = x     # Reject. Stay at x
            i += 1
            continue

        # Backward Projection
        v_prime = v_prime_sample @ ty_basis.T   # Linear combination of the basis vectors. Size: (d + m, )
        a_prime, flag, n_grad = project_zappa(manifold.q, y + v_prime, Qy, manifold.Q, tol, maxiter)
        N_EVALS['jacobian'] += n_grad
        if flag == 0:
            samples[i, :] = x     # projection failed, stay at x
            i += 1
            continue

        # Accept move x --> y
        x = y
        logfx = logfy
        samples[i, :] = x
        Qx = Qy                # Store gradient
        tx_basis = ty_basis    # Store tangent basis
        ACCEPTED[i-1] = 1
        i += 1

    return samples, N_EVALS, ACCEPTED


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