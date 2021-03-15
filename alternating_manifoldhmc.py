import numpy as np
from scipy.stats import multivariate_normal
from zappa import zappa_sampling
from gaussian_hmc import GaussianTargetHMC
from Manifolds.RotatedEllipse import RotatedEllipse
from utils import logf, logp


def AlternatingManifoldHMC(x0, n, m, Sigma, mu, T, epsilon, M, s=0.5, tol=1.48e-08, a_guess=1):
    """
    This is the most basic version of ManifoldHMC. We use 1 HMC step followed by m 
    steps of the Zappa algorithm. We do this n times so that the total number of samples 
    we obtain is n * m.
    
    x0 : Numpy Array
         Initial vector (2, ) where we start our algorithm. Equivalently, could be thought of q0.
         
    n : Int
        Number of HMC steps.
        
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
    x = x0
    samples = x
    for i_hmc in range(n):
        
        # 1 HMC step. Recall this returns [q0, q1] so we are only interested in the last one. (2, )
        x_hmc = GaussianTargetHMC(q0=x, n=1, M=M, T=T, epsilon=epsilon, Sigma=Sigma, mu=mu).sample()[-1]
        z = target.pdf(x_hmc)
        
        # m steps of Zappa (m, 2)
        samples_zappa = zappa_sampling(
            x_hmc, 
            RotatedEllipse(mu, Sigma, z), 
            logf, logp, m, s, tol, a_guess
        )
        
        # Store samples
        samples = np.vstack((samples, x_hmc))
        samples = np.vstack((samples, samples_zappa))
        #samples[i_hmc] = x_hmc
        #samples[(i_hmc+1):(i_hmc+1+m)] = samples_zappa
        
        # Last Zappa sample will be next sample
        x = samples_zappa[-1]
        
    return samples    