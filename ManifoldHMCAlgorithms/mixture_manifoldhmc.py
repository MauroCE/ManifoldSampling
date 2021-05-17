import numpy as np
from scipy.stats import multivariate_normal
from Zappa.zappa import zappa_sampling, zappa_adaptive, zappa_adaptive_returnscale, zappa_persistent
from HMC.gaussian_hmc import GaussianTargetHMC
from Manifolds.RotatedEllipse import RotatedEllipse
from utils import logf_Jacobian
from utils import logp as logp_scale
from utils import update_scale_sa
from utils import logpexp_scale
from scipy.interpolate import interp1d


def MixtureManifoldHMC(x0, alpha, N, n, m, Sigma, mu, T, epsilon, M, s=0.5, tol=1.48e-08, a_guess=1):
    """
    In this version of ManifoldHMC we use a mixture kernel. 
    With probability alpha we choose n steps of HMC, with probability (1 - alpha) we choose 
    m Zappa steps. Notice that here n is the number of HMC samples at each "iteration".
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
    logf = lambda xy: logf_Jacobian(xy, Sigma)
    logp = lambda xy: logp_scale(xy, s)
    x, z = x0, target.pdf(x0)
    samples = x
    while len(samples) < N: 

        # With probability alpha do n steps of HMC 
        if np.random.rand() <= alpha:
            new_samples = GaussianTargetHMC(q0=x, n=n, M=M, T=T, epsilon=epsilon, Sigma=Sigma, mu=mu).sample()
            
        # With probability 1 - alpha do m steps of Zappa's algorithm
        else:
            new_samples = zappa_sampling(x, RotatedEllipse(mu, Sigma, z), logf, logp, m, s, tol, a_guess)
            
        samples = np.vstack((samples, new_samples))
        x = new_samples[-1]
        z = target.pdf(x)
    return samples



def MixtureManifoldHMCPersistent(x0, alpha, N, n, m, Sigma, mu, T, epsilon, M, s=0.5, tol=1.48e-08, a_guess=1):
    """
    In this version of ManifoldHMC we use a mixture kernel. PERSISTENT VERSION OF ZAPPA.
    With probability alpha we choose n steps of HMC, with probability (1 - alpha) we choose 
    m Zappa steps. Notice that here n is the number of HMC samples at each "iteration".
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
    logf = lambda xy: logf_Jacobian(xy, Sigma)
    logp = lambda xy: logpexp_scale(xy, s)
    x, z = x0, target.pdf(x0)
    samples = x
    while len(samples) < N: 

        # With probability alpha do n steps of HMC 
        if np.random.rand() <= alpha:
            new_samples = GaussianTargetHMC(q0=x, n=n, M=M, T=T, epsilon=epsilon, Sigma=Sigma, mu=mu).sample()
            
        # With probability 1 - alpha do m steps of Zappa's algorithm
        else:
            new_samples = zappa_persistent(x, RotatedEllipse(mu, Sigma, z), logf, logp, m, s, tol, a_guess)
            
        samples = np.vstack((samples, new_samples))
        x = new_samples[-1]
        z = target.pdf(x)
    return samples




def MHMC_AdaptiveNoKernel(x0, alpha, N, n, m, Sigma, mu, T, epsilon, M, s=0.5, tol=1.48e-08, a_guess=1, ap_star=0.6):
    """
    Same as MixtureManifoldHMC but it uses zappa_adaptive rather than zappa_sampling. It does not use a kernel 
    to learn the lambda function. Hence it re-learns the correct scale at each contour.    
    """
    target = multivariate_normal(mean=mu, cov=Sigma)
    logf = lambda xy: logf_Jacobian(xy, Sigma)
    ####################################logp = lambda xy: logp_scale(xy, s)
    x, z = x0, target.pdf(x0)
    samples = x
    while len(samples) < N: 

        # With probability alpha do n steps of HMC 
        if np.random.rand() <= alpha:
            new_samples = GaussianTargetHMC(q0=x, n=n, M=M, T=T, epsilon=epsilon, Sigma=Sigma, mu=mu).sample()
            
        # With probability 1 - alpha do m steps of Zappa's algorithm
        else:
            new_samples = zappa_adaptive(x, RotatedEllipse(mu, Sigma, z), logf, m, s, tol, a_guess, ap_star, update_scale_sa)
            
        samples = np.vstack((samples, new_samples))
        x = new_samples[-1]
        z = target.pdf(x)
    return samples



def MHMC_AdaptiveKernel(x0, alpha, N, n, m, Sigma, mu, T, epsilon, M, s=0.5, tol=1.48e-08, a_guess=1, ap_star=0.6, trainevery=100):
    """
    Same as MHMC_AdaptiveNoKernel except now we use interp1d to learn the function mapping z to 
    its optimal scaling.

    trainevery : Int
              Number deciding after how many iterations we update our interpolating function. For instance
              if every=20 and by=100 this means that we train at 20, 40    
    """
    target = multivariate_normal(mean=mu, cov=Sigma)
    logf = lambda xy: logf_Jacobian(xy, Sigma)
    # Dummy interpolation function to make the algorithm work
    interp_func = lambda z: s
    x, z = x0, target.pdf(x0)
    samples = x
    # Keep an eye on how many times we choose zappa. This is so that we can learn the function.
    n_zappa = 0
    zs = []    # Store all the z values from zappa
    ss = []    # Stores all the s values from zappa
    while len(samples) < N:

        # With probability alpha do n steps of HMC
        if np.random.rand() <= alpha:
            new_samples = GaussianTargetHMC(q0=x, n=n, M=M, T=T, epsilon=epsilon, Sigma=Sigma, mu=mu).sample()

        # With probability 1 - alpha do m steps of Adaptive Zappa Algorithm. This time using interp1d
        else:
            # Run Zappa
            new_samples, s = zappa_adaptive_returnscale(x, RotatedEllipse(mu, Sigma, z), logf, m, interp_func(z), tol, a_guess, ap_star, update_scale_sa)
            # Add data to train interp1d
            zs.append(z)
            ss.append(s)
            n_zappa +=1
            # Check whether it's time to train interp1d or not
            if n_zappa == trainevery:
                ix = np.argsort(zs)
                # Bound it by 0.001 just in case we extrapolate too much
                interp_func = lambda x: np.max([
                    0.00001, 
                    interp1d(np.array(zs)[ix], np.array(ss)[ix], kind='nearest', fill_value="extrapolate")(x)
                ])
        
        samples = np.vstack((samples, new_samples))
        x = new_samples[-1]
        z = target.pdf(x)
    return samples, interp_func, n_zappa
