from audioop import mul
from codecs import raw_unicode_escape_decode
from socket import gaierror
import numpy as np
from numpy import array, exp, apply_along_axis, r_, vstack
from numpy import ones, zeros, log, diag, errstate, hstack, sqrt, pi, eye
from numpy import isfinite
from numpy.linalg import norm, solve
from numpy.random import default_rng, uniform, randn, normal

from scipy.stats import norm as ndist
from scipy.stats import beta as betadist
from scipy.optimize import fsolve
from scipy.special import gamma

from warnings import catch_warnings, filterwarnings


def f(xi): 
    """Simulator. This is a deterministic function."""
    a, b, g, k, *z = xi
    z = array(z)
    return a + b*(1 + 0.8 * (1 - exp(-g * z)) / (1 + exp(-g * z))) * ((1 + z**2)**k) * z

def f_broadcast(xi_matrix):
    """Broadcasted version of f."""
    return apply_along_axis(f, 1, xi_matrix)

def fnorm(xi, ystar):
    """This function is h(xi) = |f(xi) - y*|. Basically the function defining Chang's manifold."""
    return norm(f(xi) - ystar)

def fnorm_broadcast(xi_matrix, ystar):
    """Broadcasted version of fnorm."""
    return norm(f_broadcast(xi_matrix) - ystar, axis=1)

def data_generator(theta, N, seed):
    """Generates data with a given random seed."""
    rng = default_rng(seed)
    z = rng.normal(size=N)
    return f(r_[theta, z])

def Jf_transpose(xi):
    """Jacobian function of f."""
    _, b, g, k, *z = xi
    z = array(z)
    return vstack((
        ones(len(z)),
        (1 + 0.8 * (1 - exp(-g * z)) / (1 + exp(-g * z))) * ((1 + z**2)**k) * z,
        8 * b * (z**2) * ((1 + z**2)**k) * exp(g*z) / (5 * (1 + exp(g*z))**2),
        b*z*((1+z**2)**k)*(1 + 9*exp(g*z))*log(1 + z**2) / (5*(1 + exp(g*z))),
        diag(b*((1+z**2)**(k-1))*(((18*k + 9)*(z**2) + 9)*exp(2*g*z) + (8*g*z**3 + (20*k + 10)*z**2 + 8*g*z + 10)*exp(g*z) + (2*k + 1)*z**2 + 1) / (5*(1 + exp(g*z))**2))
    ))

def grad_fnorm(xi, ystar):
    """Gradient of h(xi)."""
    return Jf_transpose(xi) @ (f(xi) - ystar)

def logprior(xi):
    theta, z = xi[:4], xi[4:]
    with errstate(divide='ignore'):
        return log(((0 <= theta) & (theta <= 10)).all().astype('float64')) + ndist.logpdf(z).sum()

def logprior_broadcast(xi_matrix):
    with errstate(divide='ignore'):
        return ((0 <= xi_matrix[:, :4]) & (xi_matrix[:, :4] <= 10)).all(axis=1).astype('float64') + ndist.logpdf(xi_matrix[:, 4:]).sum(axis=1)

def sample_prior(n_params, n_latents):
    """Sample from prior distribution over params and latents."""
    return r_[uniform(low=0.0, high=10.0, size=n_params), randn(n_latents)]

def log_epanechnikov_kernel(xi, epsilon, fnorm, ystar):
    u = fnorm(xi, ystar)
    with errstate(divide='ignore'):
        return log((3*(1 - (u**2 / (epsilon**2))) / (4*epsilon)) * float(u <= epsilon))

def new_point(n_params, n_latents, fnorm, y_star, max_iter, epsilon, threshold):
    """Generates a new point for a given epsilon. Notice this is a different function from experiment 34."""
    func = lambda xi: r_[fnorm(xi, y_star), zeros(n_params + n_latents-1)]
    log_abc_posterior = lambda xi, epsilon: logprior(xi) + log_epanechnikov_kernel(xi, epsilon, fnorm, y_star)
    i = 0
    with catch_warnings():
        filterwarnings('error')
        while i <= max_iter:
            i += 1
            try: 
                theta_guess = uniform(0.0, 10.0, size=4)
                guess = hstack((theta_guess, zeros(n_latents)))
                point = fsolve(func, guess)
                if not isfinite([log_abc_posterior(point, epsilon)]):
                    pass
                else:
                    if fnorm(point, y_star) < threshold:
                        return point
                    else:
                        continue
            except RuntimeWarning:
                continue




class GandK:
    """This does not use fnorm."""
    def __init__(self, m=20, epsilon=2.0, parameter_prior='uniform', kernel='epanechnikov', normal_prior_scale=1.0):
        """Encapsulates various G and K functions. We always use all 4 parameters theta=(a,b,g,k).
        m : Number of latent variables
        """
        self.m = m 
        self.epsilon = epsilon
        self.ystar = None
        self.nps = normal_prior_scale
        # Check that `parameter_prior` and `kernel` have valid values
        if parameter_prior not in ['uniform', 'normal', 'beta']:
            raise NotImplementedError('Parameter prior must be uniform, normal or beta.')
        if kernel not in ['epanechnikov', 'normal']:
            raise NotImplementedError('Kernel must be epanechnikov or normal.')

        # Set correct function to sample parameter prior
        sample_theta_uniform = lambda: uniform(low=0.0, high=10.0, size=4)
        sample_theta_normal  = lambda: normal(loc=5.0, scale=self.nps, size=4)
        sample_theta_beta    = lambda: 10*betadist.rvs(a=2, b=2, scale=1, size=4)
        self.sample_theta = sample_theta_uniform
        if parameter_prior == 'normal':
            self.sample_theta = sample_theta_normal
        if parameter_prior == 'beta':
            self.sample_theta = sample_theta_beta

        # Set correct logprior function for parameters
        if parameter_prior == 'uniform':
            self.logprior           = self.logprior_uniform 
            self.logprior_broadcast = self.logprior_broadcast_uniform
        elif parameter_prior == 'normal':
            self.logprior = lambda xi: ndist.logpdf(xi[:4], loc=5.0, scale=self.nps).sum() + ndist.logpdf(xi[4:]).sum()
            self.logprior_broadcast = lambda xi_mat: ndist.logpdf(xi_mat[:, :4], loc=5.0, scale=self.nps).sum() + ndist.logpdf(xi_mat[:, 4:]).sum()
        elif parameter_prior == 'beta':
            #self.logprior = lambda xi: betadist.logpdf(xi[:4]/10, a=2, b=2, scale=1).sum() + ndist.logpdf(xi[4:]).sum()
            #self.logprior_broadcast = lambda xi_mat: betadist.logpdf(xi_mat[:, :4]/10, a=2, b=2, scale=1).sum() + ndist.logpdf(xi_mat[:, 4:]).sum()
            self.logprior = lambda xi: betadist.logpdf(xi[:4], a=2, b=2, scale=10).sum() + ndist.logpdf(xi[4:]).sum()
            #self.logprior = lambda xi: np.sum(log(3*xi[:4]) + log(10 - xi[:4]) - log(50)) +  ndist.logpdf(xi[4:]).sum()
        # Set correct kernel
        self.logkernel = self.log_epanechnikov_kernel
        if kernel == 'normal':
            self.logkernel = self.log_normal_kernel

        # Set correct dVdξ for HMC
        self.dVdξ = lambda ξ: (_ for _ in ()).throw(NotImplementedError('No dVdξ for uniform prior or non-normal kernel'))
        if parameter_prior == 'normal' and kernel == 'normal':
            self.dVdξ = self.dVdξ_normal_prior
        elif parameter_prior == 'beta' and kernel == 'normal':
            self.dVdξ = self.dVdξ_beta_prior 

    def f(self, xi):
        """Function of parameters and latent variables. xi has dimension 4 + m."""
        a, b, g, k, *z = xi
        z = array(z)
        return a + b*(1 + 0.8 * (1 - exp(-g * z)) / (1 + exp(-g * z))) * ((1 + z**2)**k) * z

    def f_broadcast(self, xi_matrix):
        """Same as GandK.f() but this is broadcasted for a matrix of xi's."""
        return apply_along_axis(self.f, 1, xi_matrix)

    def data_generator(self, theta, seed=1234):
        """Given an original theta vector, it generates z and hence produces observed data y."""
        rng = default_rng(seed)
        z = rng.normal(size=self.m)
        return self.f(r_[theta, z])

    def Jf_transpose(self, xi):
        """Returns the transpose of the Jacobian of f."""
        _, b, g, k, *z = xi
        z = array(z)
        return vstack((
            ones(len(z)),
            (1 + 0.8 * (1 - exp(-g * z)) / (1 + exp(-g * z))) * ((1 + z**2)**k) * z,
            8 * b * (z**2) * ((1 + z**2)**k) * exp(g*z) / (5 * (1 + exp(g*z))**2),
            b*z*((1+z**2)**k)*(1 + 9*exp(g*z))*log(1 + z**2) / (5*(1 + exp(g*z))),
            diag(b*((1+z**2)**(k-1))*(((18*k + 9)*(z**2) + 9)*exp(2*g*z) + (8*g*z**3 + (20*k + 10)*z**2 + 8*g*z + 10)*exp(g*z) + (2*k + 1)*z**2 + 1) / (5*(1 + exp(g*z))**2))
        ))

    def Jf(self, xi):
        """Returns Jacobian of f."""
        return self.Jf_transpose(xi).T

    def logprior_uniform(self, xi):
        """Computes log prior at xi. We assume a uniform(0, 10) prior on each parameter and a standard normal on z."""
        theta, z = xi[:4], xi[4:]
        with errstate(divide='ignore'):
            return log(((0 <= theta) & (theta <= 10)).all().astype('float64')) + ndist.logpdf(z).sum()

    def logprior_broadcast_uniform(self, xi_matrix):
        """Same as GandK.logprior() but works on a matrix of xi's."""
        with errstate(divide='ignore'):
            return ((0 <= xi_matrix[:, :4]) & (xi_matrix[:, :4] <= 10)).all(axis=1).astype('float64') + ndist.logpdf(xi_matrix[:, 4:]).sum(axis=1)

    def sample_prior(self):
        """Samples from the prior. Again uniform on theta, standard normal on z."""
        return r_[self.sample_theta(), randn(self.m)]

    def set_ystar(self, ystar):
        """Saves ystar."""
        self.ystar = ystar

    def log_epanechnikov_kernel(self, xi):
        """Log Epanechnikov kernel applied to a xi, using observed data and provided tolerance epsilon."""
        if self.ystar is None:
            raise ValueError("Set the bloody ystar, you pig.")
        u = norm(self.f(xi) - self.ystar)
        with errstate(divide='ignore'):
            return log((3*(1 - (u**2 / (self.epsilon**2))) / (4*self.epsilon)) * float(u <= self.epsilon))

    def log_normal_kernel(self, xi):
        """Log Gaussian kernel.
        For info on kernels see "Overview of ABC" in Handbook of ABC by Sisson."""
        if self.ystar is None:
            raise ValueError("Set the bloody ystar, you pig.")
        u = norm(self.f(xi) - self.ystar)
        return -u**2/(2*(self.epsilon**2)) -0.5*log(2*pi*(self.epsilon**2))

    def log_abc_posterior(self, xi):
        """Computes the ABC posterior log density."""
        return self.logprior(xi) + self.logkernel(xi)

    def V(self, xi):
        """Wrapper function for HMC."""
        return - self.log_abc_posterior(xi)

    def dVdξ_normal_prior(self, ξ):
        """Derivative for HMC for a normal prior."""
        θ, z = ξ[:4], ξ[4:]
        μ = 5*ones(4)
        Σ = (self.nps**2)*eye(4)
        dpdξ = r_[solve(Σ, θ - μ), z]
        dkdξ = self.Jf_transpose(ξ) @ (self.f(ξ) - self.ystar) / (self.epsilon**2)
        return dpdξ + dkdξ

    def dVdξ_beta_prior(self, ξ):
        """Derivative for HMC for a beta prior."""
        θ, z = ξ[:4], ξ[4:]
        #dpdξ = r_[-1/θ + 1/(10 - θ), z]
        dpdξ = r_[(-1/θ - 1/(10-θ)), z]
        dkdξ = self.Jf_transpose(ξ) @ (self.f(ξ) - self.ystar) / (self.epsilon**2)
        return dpdξ + dkdξ

    def find_point_on_manifold(self, max_iter=1000):
        # try to find a point that makes the constraint function zero..?
        i = 0
        with catch_warnings():
            filterwarnings('error')
            while i <= max_iter:
                i += 1
                try: 
                    ξ_guess = self.sample_prior()
                    ξ_found = fsolve(lambda ξ: r_[self.f(ξ) - self.ystar, zeros(4)], ξ_guess)
                    if not isfinite([self.log_abc_posterior(ξ_found)]):
                        pass
                    else:
                        return ξ_found
                        
                except RuntimeWarning:
                    continue
            raise ValueError("Couldn't find a point, try again.")

    def _find_z_given_y(self, z):
        """Instead of using the Bisection algorithm like Chang suggested, we can use `fsolve`.
        Basically given data y, we aim to find the original latent variables that gave rise to it."""
        pass

    def _bisection(self, a, b, f, tol=1e-8):
        """Bisection algorithm as devised by Chang."""
        assert a < b, "a must be smaller than b"
        assert f(a) * f(b) < 0, "f(a)f(b) must be negative"
        x = (a + b) / 2
        while abs(f(x)) >= tol:
            if f(x) * f(b) < 0:
                a = x
                x = (a + b) / 2 
            elif f(x) * f(a) < 0:
                b = x
                x = (a + b) / 2
        return x 

    # def _likelihood_calculation(self, theta, x, tol=1e-8):
    #     """Likelihood calculation a la chang."""
    #     a = 0.0
    #     b = 0.0
    #     function = lambda z: self.f(r_[theta, z]) - x
    #     while f(a) * f(b) > 0:
    #         a -= 1
    #         b += 1
    #     z = self._bisection(a, b, function, tol)
    #     return ndist.pdf(z) * 


    



    

    

    