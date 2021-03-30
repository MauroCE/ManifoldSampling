import numpy as np
from scipy.stats import multivariate_normal
from utils import logp as logp_scale
from utils import logf_Jacobian
from Zappa.zappa import zappa_sampling
from HMC.gaussian_hmc import GaussianTargetHMC
from Manifolds.RotatedEllipse import RotatedEllipse



def AlternatingManifoldHMC(x0, N, n, m, Sigma, mu, T, epsilon, M, s=0.5, tol=1.48e-08, a_guess=1):
    """
    This is the most basic version of ManifoldHMC. We use n HMC steps followed by m 
    steps of the Zappa algorithm (where the m Zappa steps start from the last HMC step).
    We do this many times until the total number of samples is N.
    
    x0 : Numpy Array
         Initial vector (2, ) where we start our algorithm. Equivalently, could be thought of q0.

    N : Int
        Total number of samples.
         
    n : Int
        Number of HMC steps per iteration.
        
    m : Int
        Number of manifold sampling steps on each contour.
    
    Sigma : Numpy Array
            Covariance matrix (2,2) of target distribution (which is Gaussian).
    mu : Numpy Array
         Mean vector (2, ) of target distribution.
         
    T : Float
        Total integration time for Leapfrog Step.
        
    epsilon : Float
              Step size for Leapfrog step.
              
    M : Numpy Array
        Covariance matrix (2, 2) of momentum distribution (which is Gaussian).
        
    s : Float
        Scale for tangent sampling of Zappa algorithm.
        
    tol : Float
          Tolerance for Zappa algorithm
          
    a_guess : Float
              Initial guess for projection in Zappa algorithm
    """
    target = multivariate_normal(mean=mu, cov=Sigma)
    logf = lambda xy: logf_Jacobian(xy, Sigma)
    logp = lambda xy: logp_scale(xy, s)
    x = x0
    samples = x
    while len(samples) < N:
        
        # n HMC step. (n, 2)
        hmc_samples = GaussianTargetHMC(q0=x, n=n, M=M, T=T, epsilon=epsilon, Sigma=Sigma, mu=mu).sample()
        x_hmc = hmc_samples[-1]  # Grab last step. This is where Zappa will start
        z = target.pdf(x_hmc)
        
        # m steps of Zappa (m, 2)
        zappa_samples = zappa_sampling(
            x_hmc, 
            RotatedEllipse(mu, Sigma, z), 
            logf, logp, m, s, tol, a_guess
        )
        
        # Store samples
        samples = np.vstack((samples, hmc_samples))
        samples = np.vstack((samples, zappa_samples[1:]))   # The last HMC sample and first Zappa sample are the same!
        
        # Last Zappa sample will be next sample
        x = zappa_samples[-1]
        
    return samples    