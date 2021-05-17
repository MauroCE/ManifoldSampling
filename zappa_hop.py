"""
This is a hybrid algorithm. It uses a Hop kernel to change energy density and it explores a contour using the adaptive Zappa algorithm.
"""
import numpy as np
from numpy.random import rand
from utils import update_scale_sa
from Zappa.zappa import zappa_adaptive
from Zappa.zappa import zappa_sampling
from HugHop.HugHopFunctions import HopKernel, HopKernelH
from scipy.stats import multivariate_normal
from Manifolds.RotatedEllipse import RotatedEllipse
from utils import logf_Jacobian
from numpy.linalg import cholesky 
from utils import logp as logp_scale


class ZappaHop:
    def __init__(self, target, alternating=False, preconditioning=False, adaptive=False):
        """
        Class combining the HOP kernel with the Zappa kernel.
        """
        # Store
        self.preconditioning = preconditioning
        self.alternating = alternating
        self.adaptive = adaptive
        # Choose whether to use Adaptive Zappa or Standard Zappa
        self.zappa = zappa_sampling
        if adaptive:
            self.zappa = zappa_adaptive

        # Choose whether to use a mixture of a cycle
        self.combination = self.mixture
        if alternating:
            self.combination = self.alternation
        
        # Choose which Hop kernel to use
        self.Hop = HopKernel
        if preconditioning:
            self.Hop = HopKernelH

        self.samples = self.grad_log_pi = self.x0 = self.s = self.m = self.tol = self.a_guess = self.maxiter = self.ap_star = None
        self.z = None
        self.update_scale = update_scale_sa
        self.l = self.k = None
        self.target = target
        self.mu, self.Sigma = self.target.mean, self.target.cov
        self.logf = lambda xy: logf_Jacobian(xy, self.Sigma)
        self.logp = lambda xy: logp_scale(xy, self.s)
        self.A = self.Sigma_hat = None

        ### ARGUMENTS FOR ZAPPA
        # Function to return arguments for standard zappa
        self.zargs = lambda: (
            self.x, RotatedEllipse(self.mu, self.Sigma, self.target.pdf(self.x)), self.logf, self.logp, self.m, self.s, self.tol, self.a_guess, self.maxiter
        )
        if adaptive:
            # Function to return arguments for adaptive zappa
            self.zargs = lambda: (
                self.x, RotatedEllipse(self.mu, self.Sigma, self.target.pdf(self.x)), self.logf, self.m, self.s, 
                self.tol, self.a_guess, self.ap_star, self.update_scale, self.maxiter
            )
        ### ARGUMENTS FOR HOP
        # Function to return arguments for standard hop
        self.hargs = lambda: (
            self.x, self.grad_log_pi, self.l, self.k, self.target.logpdf
        )
        if preconditioning:
            # Function to return arguments for preconditioned hop
            self.hargs = lambda: (
                self.x, self.grad_log_pi, self.target.logpdf, self.l, self.k, self.A, self.Sigma_hat
            )

    def sample(self, x0, grad_log_pi, N, m, s, l, k, tol=1.48e-08, a_guess=1.0, ap_star=0.6, maxiter=50):
        """
        Samples using a mixture or alternation of Hop and Zappa kernels. Based on the parameters, it will
        either use a standard Hop kernel or a preconditioned one. It will also either use a standard zappa kernel
        or an adaptive one.
        """
        # Store variables
        self.grad_log_pi, self.x, self.m, self.s, self.tol, self.a_guess, self.ap_star, self.maxiter = grad_log_pi, x0, m, s, tol, a_guess, ap_star, maxiter
        self.l, self.k = l, k
        
        # Approximate covariance matrix if preconditioninig is needed
        if self.preconditioning:
            ZappaObj = ZappaHop(self.target, self.alternating, False, self.adaptive)
            pcsamples = ZappaObj.sample(x0, grad_log_pi, N, m, s, l, k, tol, a_guess, ap_star, maxiter)
            self.Sigma_hat = np.cov(pcsamples.T)   # Covariance Approximation
            self.A = cholesky(self.Sigma_hat).T    # Transpose it because cholesky returns L s.t. LL.T = Sigma_hat, but we want A.T A = Sigma_hat

        # Construct Target
        self.x = x0
        self.z = self.target.pdf(x0)
        self.samples = self.x

        # Loop through kernels either in alternation or as a mixture
        while len(self.samples) < N:
            self.combination()
        return self.samples

    def alternation(self):
        """
        Alternation/Cycle of kernels.
        """
        # Perform 1 step of Hop kernel
        x = self.Hop(*self.hargs())
        self.samples = np.vstack((self.samples, x))

        # Perform m steps of Zappa kernel
        za_samples = self.zappa(*self.zargs())
        self.samples = np.vstack((self.samples, za_samples))
        self.x = za_samples[-1]

    def mixture(self):
        """
        Mixture of kernels.
        """
        # 1 Hop step with probability alpha
        if rand() <= alpha:
            new_samples = self.Hop(*self.hargs())
            self.x = new_samples
        
        # m Zappa steps with probability 1 - alpha
        else:
            new_samples = self.zappa(*self.zargs())
            self.x = new_samples[-1]

        # Store samples & update z
        self.samples = np.vstack((self.samples, new_samples))
        self.z = target.pdf(self.x)

    


def HopZappaMixture(x0, alpha, grad_log_pi, l, k, N, m, Sigma, mu, s, tol=1.48e-08, a_guess=1.0, ap_star=0.6):
    """
    Similar to MixtureManifoldHMC but uses Hop to change energy level rather than HMC.
    """
    # Construct the target
    target = multivariate_normal(mu, Sigma)
    logf = lambda xy: logf_Jacobian(xy, Sigma)
    x, z = x0, target.pdf(x0)
    samples = x
    while len(samples) < N: 

        # With probability alpha do 1 Hop step
        if rand() <= alpha:
            new_samples = HopKernel(x, grad_log_pi, l, k, target.logpdf)
            x = new_samples   # There is only one!

        # With probability 1 - alpha do m adaptive zappa steps
        else:
            new_samples = zappa_adaptive(x, RotatedEllipse(mu, Sigma, z), logf, m, s, tol, a_guess, ap_star, update_scale_sa)
            x = new_samples[-1]

        samples = np.vstack((samples, new_samples))
        z = target.pdf(x)

    return samples



def HopZappaMixturePC(x0, alpha, grad_log_pi, l, k, N, m,  Sigma, mu, s, burnin, tol=1.48e-08, a_guess=1.0, ap_star=0.6):
    """
    Same as HopZappaMixture but now Hop kernel uses preconditioning. In order to use preconditioning it rus HopZappaMixture 
    for burnin iterations and then computes an empirical covariance matrix and finds the transpose of its cholesky decomposition.
    """
    # Burnin, find covariance matrix and transpose of its cholesky factor
    hhsamples = HopZappaMixture(x0, alpha, grad_log_pi, l, k, burnin, m, Sigma, mu, s, tol, a_guess, ap_star)
    Sigma_hat = np.cov(hhsamples.T)   # Covariance Approximation
    A = cholesky(Sigma_hat).T         # Transpose it because cholesky returns L s.t. LL.T = Sigma_hat, but we want A.T A = Sigma_hat

    # Construct target
    target = multivariate_normal(mu, Sigma)
    logf = lambda xy: logf_Jacobian(xy, Sigma)
    x, z = x0, target.pdf(x0)
    samples = x
    while len(samples) < N:

        # With probability alpha do 1 Hessian Hop Step
        if rand() <= alpha:
            new_samples = HopKernelH(x, grad_log_pi, target.logpdf, l, k, A, Sigma_hat)
            x = new_samples

        # With probability 1 - alpha do m adaptive zappa steps
        else:
            new_samples = zappa_adaptive(x, RotatedEllipse(mu, Sigma, z), logf, m, s, tol, a_guess, ap_star, update_scale_sa)
            x = new_samples[-1]

        samples = np.vstack((samples, new_samples))
        z = target.pdf(x)
    return samples



def HopZappaAlternating(x0, grad_log_pi, l, k, N, m, Sigma, mu, s, tol=1.48e-08, a_guess=1.0, ap_star=0.6):
    """
    Similar to HopZappaMixture but rather than being a mixture of Hop and AdaptiveZappa kernels, it 
    alternates them.
    """
    # Construct target
    target = multivariate_normal(mu, Sigma)
    logf = lambda xy: logf_Jacobian(xy, Sigma)
    x = x0
    samples = x
    while len(samples) < N:

        # 1 step of Hop Kernel
        x = HopKernel(x, grad_log_pi, l, k, target.logpdf)
        samples = np.vstack((samples, x))

        # m steps of adaptive zappa
        za_samples = zappa_adaptive(x, RotatedEllipse(mu, Sigma, target.pdf(x)), logf, m, s, tol, a_guess, ap_star, update_scale_sa)
        samples = np.vstack((samples, za_samples))
        x = za_samples[-1]

    return samples


def HopZappaAlternatingPC(x0, grad_log_pi, l, k, N, m, Sigma, mu, s, burnin, tol=1.48e-08, a_guess=1.0, ap_star=0.6):
    """
    Same as HopZappaAlternating but uses preconditioning. It runs HopZappaAlternating for burnin iterations and then computes
    an estimate for the covariance matrix and the transpose of its cholesky factor.
    """
    # Burnin
    hhsamples = HopZappaAlternating(x0, grad_log_pi, l, k, burnin, m, Sigma, mu, s, tol, a_guess, ap_star)
    Sigma_hat = np.cov(hhsamples.T)   # Covariance Approximation
    A = cholesky(Sigma_hat).T         # Transpose it because cholesky returns L s.t. LL.T = Sigma_hat, but we want A.T A = Sigma_hat

    # Construct target
    target = multivariate_normal(mu, Sigma)
    logf = lambda xy: logf_Jacobian(xy, Sigma)
    x, z = x0, target.pdf(x0)
    samples = x

    while len(samples) < N:

        # 1 step of Hop Kernel
        x = HopKernelH(x, grad_log_pi, target.logpdf, l, k, A, Sigma_hat)
        samples = np.vstack((samples, x))

        # m steps of adaptive zappa
        za_samples = zappa_adaptive(x, RotatedEllipse(mu, Sigma, target.pdf(x)), logf, m, s, tol, a_guess, ap_star, update_scale_sa)
        samples = np.vstack((samples, za_samples))
        x = za_samples[-1]

    return samples