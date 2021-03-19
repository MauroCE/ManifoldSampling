import numpy as np
from scipy.stats import multivariate_normal
from zappa import zappa_sampling
from gaussian_hmc import GaussianTargetHMC
from Manifolds.RotatedEllipse import RotatedEllipse
from utils import logf, logp



def MixtureManifoldHMC(x0, alpha, N, n, m, Sigma, mu, T, epsilon, M, s=0.5, tol=1.48e-08, a_guess=1):
    """
    In this version of ManifoldHMC we use a mixture kernel. With probability alpha we choose
    1 step of HMC, with probability (1 - alpha) we choose m Zappa steps.
    IMPORTANT: Notice that here n is the number of HMC samples at each "iteration".
    Total number of samples is N.
    
    x0 : Numpy Array
         Initial vector (2, ) where we start our algorithm. Equivalently, could be thought of q0.

    alpha : Float
            Probability (must be between 0 and 1) of using the HMC kernel. 1 - alpha is the probability
            of using the Zappa kernel.

    N : Int
        Total number of samples
         
    n : Int
        Number of HMC samples at each iteration.
        
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
    x, z = x0, target.pdf(x0)
    samples = x
    while len(samples) < N: 

        # With probability alpha do n steps of HMC 
        if np.random.rand() <= alpha:
            new_samples = GaussianTargetHMC(q0=x, n=n, M=M, T=T, epsilon=epsilon, Sigma=Sigma, mu=mu).sample()  #[1:]
            
        # With probability 1 - alpha do m steps of Zappa's algorithm
        else:
            new_samples = zappa_sampling(x, RotatedEllipse(mu, Sigma, z), logf, logp, m, s, tol, a_guess)
            
        samples = np.vstack((samples, new_samples))
        x = new_samples[-1]
        z = target.pdf(x)
    return samples
