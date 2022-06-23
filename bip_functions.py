from numpy import array, log, zeros, eye, errstate
from numpy.linalg import norm

from scipy.stats import multivariate_normal as MVN

### Functions for standard BIP-posterior

def F(θ):
    """Deterministic function."""
    return array([θ[1]**2 + 3 * θ[0]**2 * (θ[0]**2 - 1)])

def F_broadcast(θ_matrix):
    """Broadcasted version of F."""
    return θ_matrix[:, 1]**2 + (3*θ_matrix[:, 0]**2)*(θ_matrix[:, 0]**2 - 1)

def grad_F(θ):
    """Gradient of Deterministic Function."""
    return array([12*θ[0]**3 - 6*θ[0], 2*θ[1]])

def logprior(θ):
    """Log-density of prior on θ."""
    return MVN(zeros(2), eye(2)).logpdf(θ)

def sample_prior():
    """Samples from the prior."""
    return MVN(zeros(2), eye(2)).rvs()

def grad_logprior(θ):
    """Gradient of Log prior. Since it's a standard normal, it's simply -θ."""
    return -θ

def log_posterior(θ, y, σ):
    """Log-density of posterior of θ given observation."""
    return logprior(θ) - norm(y - F(θ))**2 / (2*σ**2) - 1*log(σ)

def grad_logpost(θ, y, σ):
    """Gradient of log-density of posterior."""
    return grad_logprior(θ) + (y - F(θ))*grad_F(θ) / (σ**2)


### Functions for Approximate Lifted Posterior

def FL(ξ, σ):
    """Deterministic function of θ and η."""
    return F(ξ[:2])[0] + σ * ξ[-1]

def FL_broadcast(ξ, σ):
    """Broadcasted version of FL."""
    return F_broadcast(ξ[:, :2]) + σ * ξ[:, -1]

def logpriorL(ξ):
    """Logprior for approximate lifted distribution."""
    return MVN(zeros(3), eye(3)).logpdf(ξ)

def sample_priorL():
    """Samples from prior"""
    return MVN(zeros(3), eye(3)).rvs()

def log_epanechnikov_kernelL(ξ, ϵ, σ, y):
    """Kernel for approximate lifted distribution."""
    u = norm(FL(ξ, σ) - y)
    with errstate(divide='ignore'):
        return log((3*(1 - (u**2 / (ϵ**2))) / (4*ϵ)) * float(norm(FL(ξ, σ) - y) <= ϵ))

def log_posteriorL(ξ, ϵ, σ, y):
    """Approximate lifted distribution."""
    return logpriorL(ξ) + log_epanechnikov_kernelL(ξ, ϵ, σ, y)

def grad_FL(ξ, σ):
    """Gradient of the function Gσ with respect to """
    return array([12*ξ[0]**3 - 6*ξ[0], 2*ξ[1], σ])

def hess_FL(ξ, σ):
    """Hessian of the function."""
    return array([
        [36*ξ[0]**2-6, 0, 0],
        [0, 2, 0],
        [0, 0, 0]
    ])

