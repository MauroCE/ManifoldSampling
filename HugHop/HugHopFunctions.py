import numpy as np
from numpy import log, sqrt
from numpy import eye, outer, zeros
from numpy.random import rand
from scipy.stats import multivariate_normal
from numpy.linalg import cholesky, inv, eigh, cholesky, solve, det, norm



def HopKernel(x, grad_log_pi, l, k, logpi):
    """
    Standard Hop Kernel with no preconditioning. This is ONE STEP of the hop kernel.

    x : Numpy Array
        Point form which to perform a Hop.

    grad_log_pi : Callable
                  Gradient of log target density. E.g. for multivariate normal would be
                  lambda xy: - inv(Sigma) @ xy

    l : Float
        Lambda. Controls scaling parallel to the gradient.

    k : Float 
        We have mu^2 = k * l where mu^2 controls scaling perpendicular to gradient.

    logpi : Callable
            Log target density. E.g. for MVN would be multivariate_normal().logpf
    """       
    # Find parameters
    mu_sq = k * l
    mu = sqrt(mu_sq)
    l_sq = sqrt(l)
    d = len(x)
    
    # For MH step
    logu = log(rand())
    
    # Gradient, its norm and nornmalized gradient
    gx = grad_log_pi(x)
    gx_norm = norm(gx)
    gxhat = gx / gx_norm
    
    # B^{1/2}
    B_sqrt = (mu*eye(d) + (l - mu) * outer(gxhat, gxhat)) / sqrt(1 + gx_norm**2)
    
    # Sample 
    v = multivariate_normal(mean=zeros(d), cov=eye(d)).rvs()
    
    # Proposal
    y = x + (B_sqrt @ v)
    
    # Compute gradient stuff at y
    gy = grad_log_pi(y)
    gy_norm = norm(gy)
    gyhat = gy / gy_norm
    
    # Accept-Reject
    logr = logpi(y) - logpi(x) + (d/2) * log((gy_norm**2) / (gx_norm**2))
    logr = logr - (1/(2*mu_sq)) * (norm(y-x)**2)*((gy_norm**2) - (gx_norm**2))
    logr = logr - 0.5*((1/l_sq) - (1/mu_sq)) * (((y - x) @ gy)**2 - ((y - x) @ gx)**2)
    if logu <= min(0, logr):
        # Accept
        return y#, 1
    else:
        return x#, 0



def HopKernelH(x, grad_log_pi, logpi, l, k, A, Sx):
    """
    Hop Kernel using preconditioninig. This is ONE STEP of the hop kernel.

    x : Numpy Array
        Point form which to perform a Hop.
    """        
    # Find parameters
    mu_sq = k * l
    mu = sqrt(mu_sq)
    l_sq = sqrt(l)
    d = len(x)

    # For MH step
    logu = log(rand())
    
    # Gradient, its norm and nornmalized gradient
    gx = grad_log_pi(x)
    gx_norm = norm(gx)
    gxhat = gx / gx_norm
    gtx = A @ gx   # g tilde x
    gtx_norm = norm(gtx)
    
    
    # Denominator
    denom = gx @ Sx @ gx
    
    # B
    B = (mu_sq*Sx + (l_sq - mu_sq) * (Sx @ outer(gx, gx) @ Sx.T) / denom) / denom
    B_sqrt = cholesky(B)
    
    # Hessian
    H = -inv(Sx)
    
    # Sample 
    v = multivariate_normal(mean=zeros(d), cov=eye(d)).rvs()
    
    # Proposal
    y = x + (B_sqrt @ v)
    
    # Compute gradient stuff at y
    gy = grad_log_pi(y)
    gy_norm = norm(gy)
    gyhat = gy / gy_norm
    gty = A @ gy
    gty_norm = norm(gty)
    xmy_norm = norm(x - y) # ||x - y||
    
    # Accept-Reject
    # I am assuming S(x) = S(y) = S hence 0.5*log(det(S(x)) / det(S(y))) doesn't appear
    logr = logpi(y) - logpi(x) + (d/2) * log((gty_norm**2) / (gtx_norm**2)) 
    logr = logr - (1/(2*mu_sq)) * ((y-x) @ ((gtx_norm**2)*H - (gty_norm**2)*H) @ (y - x))
    logr = logr - 0.5*((1/l_sq) - (1/mu_sq)) * (((y - x) @ gy)**2 - ((y - x) @ gx)**2)
    if logu <= min(0, logr):
        # Accept
        return y#, 1
    else:
        return x#, 0