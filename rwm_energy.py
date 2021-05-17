import numpy as np 
from numpy.linalg import inv, norm
from numpy.random import randn, rand
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal

from utils import logp as logp_scale
from Zappa.zappa import zappa_sampling
from Zappa.ZappaAdaptive import ZappaAdaptive
from Manifolds.RotatedEllipse import RotatedEllipse
from utils import logf_Jacobian, quick_MVN_marginals, quick_MVN_scatter


def get_scaling_function(num, target, logf, n=200, s0=0.5, tol=1.48e-08, a_guess=1.0, ap_star=0.6, exponent=(2/3)):
    """
    Learns correct scaling for each contour.
    """
    # Run it num times and store z and s values
    z_vals = []
    s_vals = []
    for _ in range(num):
        # Sample a point from the target
        x = target.rvs()
        z = target.pdf(x)
        # Sample on its contour
        ellipse = RotatedEllipse(target.mean, target.cov, z)
        ZappaObj = ZappaAdaptive(x, ellipse, logf, logp_scale, s0, n, tol, a_guess, lambda x: x, ap_star, exponent)
        out = ZappaObj.sample()
        # Save its optimal scaling
        z_vals.append(z)
        s_vals.append(np.exp(out['LogScales'][-1]))
    # Learn from dataset using interp1d. Extrapolate to avoid errors
    return interp1d(z_vals, s_vals, kind="nearest", fill_value="extrapolate")


class RWEnergy:
    def __init__(self, x, target, contour_func, scale_func, glp, niter, n, scaling, clipval=1.0, lbval=None, clip=True, 
    reversecheck=False, tol=1.48e-08, a_guess=1.0):
        """Uses a random walk to explore the energy distribution. This requires us to estimate normalizing constants
        which we do via importance sampling on the contour.
        
        x : np.array
            Starting point of the algorihm.
        target : Object
                 Target distribution. Must have methods .pdf and .logpdf. At the moment this only works for MVN.
        contour_func: callable
                      Function that takes in a value z and returns a contour. For instance for MVN would be 
                      `lambda z: RotatedEllipse(mu, Sigma, z)`.
        scale_func: callable
                    Learned function that maps z to optimal scaling. Should be the output of `get_scaling_function`.
        glp : callable
              "Grad Log Pi". This should be a function that takes a set of samples (n, d) and computes the gradient of
              log target for each of them in parallel. For instance for a MVN glp would be (xy - mu) @ inv(Sigma).T.
        niter : int
                Number of energy iterations we would like to achieve.
        n : int
            Number of samples per energy level.
        scaling: float
                 Scaling for sampling a new energy. At the moment we are using a gaussian `u + scaling * randn()`.
        lbval : float
                Lower bound value for the norm. This is used only when clip=False.
        clip : bool
               Whether to clip weights or not. If weights are not clipped we LOWER BOUND the norm of the gradient.
        clipval : float
                  Maximum value allowed for each individual weights. This is needed to allow exploding weights.
        reversecheck: bool
                      After proposing a new energy `ucand` from `u` if reversecheck=True we will use root-finding to try
                      and see if we could have proposed `u` from `ucand`. 
        tol : float
              Tolerance for Zappa algorithm.
        a_guess: float
                 Initial guess for `a` in Zappa.
        """
        # Store variables
        self.x = x
        self.target = target
        self.contour_func = contour_func
        self.scale_func = scale_func
        self.niter = niter
        self.n = n
        self.scaling = scaling
        self.clip = clip
        self.clipval = clipval
        self.tol = tol
        self.a_guess = a_guess
        self.glp = glp
        self.lbval = lbval
        self.reversecheck = reversecheck
        
        # Variables to store things
        self.energies = []
        self.samples, self.weights = None, None
        self.normweights = None
        self.normalizing_constants = []
        self.failed_rootfindings = 0
        self.failed_reverserootfindings = 0
        self.rejected_energies = []
        self.energies_long = None
        self.rejected_energies_reverse = []

        # Check that we have lower bound value when we need it
        if not clip and lbval is None:
            raise ValueError("You must provide `lbval` when `clip` is False.") 

        # Functions
        self.U = lambda xy: -self.target.logpdf(xy)          # Potential Function
        self.logf = lambda xy: 0                        # Uniform target on contour

        # Choose correct function to compute weights
        self.compute_weights = self._compute_clipped_weights
        if not clip:
            self.compute_weights = self._compute_normbounded_weights

        # Choose correct reverse check function
        self.check_reverse = lambda u, xcand, xu, contour, logp, s, zu, ucand: True   # Empty function by default, does nothing
        if reversecheck:
            self.check_reverse = self._check_reverse

    def sample(self):
        """
        Samples from the target.
        """
        # Sample on initial contour. Compute weights & approx normalizing constant
        u = self.U(self.x)
        z = self.target.pdf(self.x)
        s = self.scale_func(z)
        logp = lambda xy: logp_scale(xy, s)
        contour = self.contour_func(z)
        xu = zappa_sampling(self.x, contour, self.logf, logp, self.n, s, self.tol, self.a_guess)
        wu = self.compute_weights(xu)
        wunorm = wu / np.sum(wu)
        zu = np.mean(wu)

        # Housekeeping
        self.energies.append(u)
        self.samples = xu 
        self.weights = wu
        self.normweights = wunorm
        self.normalizing_constants.append(zu)
        self.energies_long = np.repeat(u, self.n)

        # Algorithm runs until we have explored `niter` energies.
        while len(self.energies) < self.niter:

            # Propose new energy candidate. This defines a new contour
            ucand = u + self.scaling * randn()

            # Attempt to find a point on the new contour. Starting point is the last sample on previous contour.
            out = root(lambda x: np.array([self.U(x) -ucand, 0]), xu[-1])

            # If root-finding failed, we stay at the current location. 
            # Since we stay at the same contour, we sample new points on it
            # and improve our normalizing constant estimate.
            if not out.success:

                # Sample again on the contour & update normalizing constant
                xu = zappa_sampling(xu[-1], contour, self.logf, logp, self.n, s, self.tol, self.a_guess)  # Sample again. Start from last point on this contour. Same optimal scaling.
                wu = self.compute_weights(xu)                                                      # Compute new weights
                wunorm = wu / np.sum(wu)                                                       # Normalize weights
                zu = (zu + np.mean(wu)) / 2                                                          # Update normalizing constant. TODO: Check wu, wunew have same length.

                # Store the new samples, weights & (same) energy
                self.weights = np.hstack((self.weights, wu))
                self.samples = np.vstack((self.samples, xu))
                self.normweights = np.hstack((self.normweights, wunorm))
                self.energies.append(u)
                self.energies_long = np.hstack((self.energies_long, np.repeat(u, self.n)))
                self.normalizing_constants.append(zu)
                self.failed_rootfindings += 1
                self.rejected_energies.append(ucand)
                continue
            # If root-finding successful, store solution. This is a point on the candidate contour.
            xcand = out.x
            if not self.check_reverse(u, xcand, xu, contour, logp, s, zu, ucand):   # Do reverse check
                continue
            ucand = self.U(xcand)                                     # u-value on new contour
            zcand = self.target.pdf(xcand)                            # z-value on new contour
            scand = self.scale_func(zcand)                            # Optimal scaling on new contour
            logpcand = lambda xy: logp_scale(xy, scand)               # Proposal function with new optimal scaling
            contourcand = self.contour_func(zcand)                    # New contour

            # Sample on new contour. Start from root-finding solution. Use optimal scaling & uniform target
            xucand = zappa_sampling(xcand, contourcand, self.logf, logpcand, self.n, scand, self.tol, self.a_guess)

            # Unnormalized, normalized weights & approx to normalizing constant of new contour
            wucand = self.compute_weights(xucand) 
            wucandnorm = wucand / np.sum(wucand)
            zucand = np.mean(wucand) 

            # Metropolis-Hastings
            logrhat = np.log(zucand) - np.log(zu) + u - ucand
            if np.log(rand()) < logrhat: # min(1, rhat):
                # Accept candidate! Now update all the variables
                u = ucand
                z = zcand
                s = scand
                logp = logpcand
                contour = contourcand
                xu = xucand
                wu = wucand
                wunorm = wucandnorm
                zu = zucand

            else:
                # Stay on the current contour. As for unsuccessful root-finding, 
                # use this opportunity to improve contour exploration.
                # Sample again on the contour & update normalizing constant
                xu = zappa_sampling(xu[-1], contour, self.logf, logp, self.n, s, self.tol, self.a_guess)  # Sample again. Start from last point on this contour. Same optimal scaling.
                wu = self.compute_weights(xu)      # Compute new weights
                wunorm = wu / np.sum(wu)           # Normalize weights
                zu = (zu + np.mean(wu)) / 2        # Update normalizing constant. TODO: Check wu, wunew have same length.
                                                     
            # Store the new samples, weights & (same) energy
            self.energies.append(u)
            self.energies_long = np.hstack((self.energies_long, np.repeat(u, self.n)))
            self.samples = np.vstack((self.samples, xu))
            self.weights = np.hstack((self.weights, wu))
            self.normweights = np.hstack((self.normweights, wunorm))
            self.normalizing_constants.append(zu)
        return self.samples, self.energies, self.weights, self.normweights, self.failed_rootfindings, self.normalizing_constants, self.rejected_energies, self.energies_long, self.failed_reverserootfindings
        
    def _compute_clipped_weights(self, samples):
        """Computes the weights using clipping."""
        return np.clip(1 / norm(self.glp(samples), axis=1), a_min=None, a_max=self.clip)
    
    def _compute_normbounded_weights(self, samples):
        """Computes weights but lower bounds the norm of the gradient."""
        return 1 / (norm(self.glp(samples), axis=1) + self.lbval)

    def _check_reverse(self, u, xcand, xu, contour, logp, s, zu, ucand):
        """Checks whether we would reach contour `ucand` from `u`."""
        out = root(lambda x: np.array([self.U(x) -u, 0]), xcand)
        if not out.success:
            # Sample again on the contour & update normalizing constant
            xunew = zappa_sampling(xu[-1], contour, self.logf, logp, self.n, s, self.tol, self.a_guess)  # Sample again. Start from last point on this contour. Same optimal scaling.
            wunew = self.compute_weights(xunew)                                            # Compute new weights
            wunewnorm = wunew / np.sum(wunew)                                        # Normalize weights
            zu = (zu + np.mean(wunew)) / 2                                           # Update normalizing constant. TODO: Check wu, wunew have same length.
            # Store the new samples, weights & (same) energy
            self.weights = np.hstack((self.weights, wunew))
            self.samples = np.vstack((self.samples, xunew))
            self.normweights = np.hstack((self.normweights, wunewnorm))
            self.energies.append(u)
            self.energies_long = np.hstack((self.energies_long, np.repeat(u, self.n)))
            self.normalizing_constants.append(zu)
            self.failed_reverserootfindings += 1
            self.rejected_energies_reverse.append(ucand)
            return False     # Meaning while loop will "continue"
        return True

    def approx_expectation(self, func):
        """Approximates expectations using the samples."""
        means = []
        for i in range(self.niter):
            means.append(
                np.sum(
                    func(self.samples[i*self.n:(i*self.n + self.n)]) * self.normweights[i*self.n:(i*self.n + self.n)][:, None],
                    axis=0
                )
            )
        return np.mean(np.vstack(means), axis=0)

