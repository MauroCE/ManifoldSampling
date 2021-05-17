import numpy as np
from numpy.random import randn, rand
from numpy.linalg import svd
from numpy import log, zeros
import matplotlib.pyplot as plt
from numpy import pi
from scipy.optimize import root
from scipy.stats import multivariate_normal, norm
from utils import normalize
from scipy.optimize.minpack import _root_hybr
import scipy.linalg as la


class ZappaAdaptive:

    def __init__(self, x0, manifold, logf, logp_scale, s0, n, tol, a_guess, K, ap_star, exponent, polyak=False, maxiter=50):
        """
        Zappa sampling but stores everything.
        """
        # Choose root-finding function
        self.project = self.project_original
        # Choose S.A. algorithm
        self.update_scale = self.update_scale_original
        if polyak:
            self.update_scale = self.update_scale_polyak

        # Store variables
        self.x0 = x0
        self.manifold = manifold
        self.logf = logf
        self.logp_scale = logp_scale
        self.logp = lambda xy: logp_scale(xy, self.s)
        self.s = s0
        self.l = np.log(self.s)
        self.K = K     # Kernel
        self.u = -np.log(self.manifold.z)
        self.ap_star = ap_star   # Optimal acceptance probability
        self.exponent = exponent
        self.n = int(n)
        self.tol = tol
        self.a_guess = a_guess
        self.maxiter = maxiter
        self.d = self.manifold.get_dimension()
        self.m = self.manifold.get_codimension()
        self.logu = log(rand(self.n))

        # Instantiate variable and arrays used to store things
        self.samples = zeros((self.n, self.d + self.m))
        self.nfevxs = zeros(self.n)                             # Number of function evaluations when projecting x + v to find y
        self.nfevys = zeros(self.n)                             # Number of function evaluations when projecting y + v' to find x
        self.njevxs = zeros(self.n)                             # Number of Jacobian evaluations when projecting x + v to find y
        self.njevys = zeros(self.n)                             # Number of Jacobian evaluations when projecting y + v' to find x
        self.statusesx = zeros(self.n)                          # Stores optimization status number for x + v -> y
        self.statusesy = zeros(self.n)                          # Stores optimization status number for y + v' -> x
        self.vxs = zeros((self.n, self.d + self.m))             # Velocities sampled at x
        self.vys = zeros((self.n, self.d + self.m))             # Velocities sampled at y
        self.flagxs = zeros(self.n)                             # Flags (1 = Success, 0 = Fail) for reprojection x + v -> y
        self.flagys = zeros(self.n)                             # Flags (1 = Success, 0 = Fail) for reprojection y + v' -> x
        self.logaps = zeros(self.n)                             # log acceptance probabilities for MH step.
        self.axs = zeros(self.n)                                # a Values for projection x + v -> y
        self.ays = zeros(self.n)                                # a' Values for projection y + v' -> x
        self.gxs = zeros((self.n, self.d + self.m))             # Stores gradient basis at x
        self.gys = zeros((self.n, self.d + self.m))             # Stores gradient basis at y
        self.txs = zeros((self.n, self.d + self.m))             # Stores tangent basis at x
        self.tys = zeros((self.n, self.d + self.m))             # Stores tangent basis at y
        self.v_samples = zeros(self.n)                          # Stores v sample (for x)
        self.v_prime_samples = zeros(self.n)                    # Stores v' sample (for y)
        self.ys = zeros((self.n, self.d + self.m))              # Stores all ys
        self.Qxs = zeros((self.n, self.d + self.m))             # Stores all the gradients Qx
        self.Qys = zeros((self.n, self.d + self.m))             # Stores all the gradients Qy
        self.ls = [self.l]                                      # Stores log scales
        self.avg_ls = [self.l]                                  # Stores Polyak averaging terms

        # Initiate algorithm
        self.x = self.x0                                        # Initial point on manifold
        #self.samples[0, :] = self.x
        self.i = 0

        # Variables that will be checked and stored
        self.vx, self.vy = None, None
        self.flagx, self.flagy = None, None
        self.nfevx, self.nfevy = None, None
        self.njevx, self.njevy = None, None
        self.statusx, self.statusy = None, None
        self.log_ap = -np.inf
        self.ax, self.ay = None, None
        self.gx_basis = self.tx_basis = self.gy_basis = self.ty_basis = np.array([None, None])
        self.v_sample = self.v_prime_sample = None
        self.y = None
        self.Qx = self.Qy = np.array([None, None])

        # Store events
        self.events = []


    def end_iteration(self, event):
        """
        Stores the sample and does a lot of housekeeping.
        """
        # Store sample
        self.samples[self.i, :] = self.x
        # Store velocities
        self.vxs[self.i, :] = self.vx
        self.vys[self.i, :] = self.vy
        # Store flags / successes
        self.flagxs[self.i] = self.flagx
        self.flagys[self.i] = self.flagy
        # Store number of function evaluations
        self.nfevxs[self.i] = self.nfevx
        self.nfevys[self.i] = self.nfevy
        # Store number of jacobian evaluations
        self.njevxs[self.i] = self.njevx
        self.njevys[self.i] = self.njevy
        # Store `a` for projections
        self.axs[self.i] = self.ax
        self.ays[self.i] = self.ay
        # Store statuses
        self.statusesx[self.i] = self.statusx
        self.statusesy[self.i] = self.statusy
        # Store log acceptance probability
        self.logaps[self.i] = self.log_ap
        # Store gradient and tangent basis
        self.gxs[self.i, :] = self.gx_basis.flatten()
        self.gys[self.i, :] = self.gy_basis.flatten()
        self.txs[self.i, :] = self.tx_basis.flatten()
        self.tys[self.i, :] = self.ty_basis.flatten()
        # Store v_sample and v_sample_prime
        self.v_samples[self.i] = self.v_sample
        self.v_prime_samples[self.i] = self.v_prime_sample
        # Stores y
        self.ys[self.i, :] = self.y
        # Store Qx and Qy
        self.Qxs[self.i, :] = self.Qx.flatten()
        self.Qys[self.i, :] = self.Qy.flatten()
        # Update the scaling factor! If we have arrived at the MH step then we will have an actual log alpha.
        # If not, then we will just consider alpha = 0 and log_ap = -inf
        self.update_scale()

        # Reset all of them to None, except x
        self.y = self.vx = self.vy  = self.nfevx = self.nfevy = self.njevx = self.njevy = None
        self.flagx = self.flagy = self.ax = self.ay = self.statusx = self.statusy = None
        self.gx_basis = self.gy_basis = self.tx_basis = self.ty_basis = np.array([None, None])
        self.v_sample = self.v_prime_sample = None
        self.Qx = self.Qy = np.array([None, None])
        self.log_ap = -np.inf

        # Increase iteration counter
        self.i += 1

        # Store list of events
        self.events.append(event)

    def prepare_output(self):
        """
        Prepares the output so that it is easy to access.
        """
        out = {
            'samples': self.samples,
            'vxs': self.vxs,
            'vys': self.vys,
            'flagxs': self.flagxs,
            'flagys': self.flagys,
            'nfevxs': self.nfevxs,
            'nfevys': self.nfevys,
            'njevxs': self.njevxs,
            'njevys': self.njevys,
            'axs': self.axs,
            'ays': self.ays,
            'statusesx': self.statusesx,
            'statusesy': self.statusesy,
            'logaps': self.logaps,
            'events': np.array(self.events),
            'gxs': self.gxs,
            'gys': self.gys,
            'txs': self.txs,
            'tys': self.tys,
            'vsamples': self.v_samples,
            'vprimesamples': self.v_prime_samples,
            'ys': self.ys,
            'Qxs': self.Qxs,
            'Qys': self.Qys,
            'LogScales': self.ls,
            'avg_ls': self.avg_ls
        }
        return out

    def sample(self):
        """
        Samples from logf on manifold using Zappa.
        """
        # Run until you get n samples
        while self.i < self.n:

            # Compute gradient, gradient basis & tangent basis at x
            self.Qx = self.manifold.Q(self.x)                            # Gradient at x.                             Size: (d + m, )
            self.gx_basis = normalize(self.Qx)                           # ON basis for gradient at x                 Size: (d + m, )
            self.tx_basis = self.manifold.tangent_basis(self.Qx)         # ON basis for tangent space at x using SVD  Size: (d + m, d)

            # Sample along tangent 
            self.v_sample = self.s*randn(self.d)                # Isotropic MVN with scaling s         Size: (d, )
            self.vx = self.tx_basis @ self.v_sample                      # Linear combination of the basis vectors  Size: (d + m, )

            # Forward Projection
            self.ax, self.flagx, self.nfevx, self.njevx, self.statusx = self.project(self.x, self.vx, self.Qx)
            # Projection has failed:
            if self.flagx == 0:                                      # Projection failed
                self.end_iteration("ProjFailed")
                continue
            # Projection is successful
            self.y = self.x + self.vx + self.ax*self.Qx.flatten()   # Projected point (d + m, )

            # Compute v' from y
            self.Qy = self.manifold.Q(self.y)                         # Gradient at y.                            Size: (d + m, )
            self.gy_basis = normalize(self.Qy)             # ON basis for gradient at y                Size: (d + m, )
            self.ty_basis = self.manifold.tangent_basis(self.Qy)      # ON basis for tangent space at y using SVD Size: (d + m, d)
            self.v_prime_sample = (self.x - self.y) @ self.ty_basis  # Components along tangent                  Size: (d + m, )

            # Metropolis-Hastings
            self.log_ap = self.logf(self.y) + self.logp(self.v_prime_sample) - self.logf(self.x) - self.logp(self.v_sample)
            # Stochastic Approximation update
            #self.update_scale()
            if self.logu[self.i] > self.log_ap:
                self.end_iteration("MHRejection")
                continue

            # Backward Projection
            self.vy = self.v_prime_sample @ self.ty_basis.T   # Linear combination of the basis vectors. Size: (d + m, )
            self.ay, self.flagy, self.nfevy, self.njevy, self.statusy = self.project(self.y, self.vy, self.Qy)
            # Re-projection has failed
            if self.flagy == 0:
                self.end_iteration("ReprojFailed")
                continue

            # Everything has succeded! Accept move
            self.x = self.y
            self.end_iteration("Success")
        return self.prepare_output()

    def update_scale_original(self):
        """Updates the scale using the normal iterative scheme"""
        step_size = 1 / (self.i + 1) ** self.exponent
        self.l += step_size * (np.exp(self.log_ap) - self.ap_star)
        self.ls.append(self.l)
        self.s = np.exp(self.l)

    def update_scale_polyak(self):
        """Updates scale but then uses the averaged version."""
        # Compute new log scale value
        step_size =  1 / (self.i + 1)**self.exponent
        new_l = self.l + step_size * (np.exp(self.log_ap) - self.ap_star)
        # Polyak averaging for log scale
        new_avg = self.avg_ls[-1] + (1 / (len(self.avg_ls) + 1)) * (new_l - self.avg_ls[-1])
        # Store the new l value. These values are only stored to see if they diverge 
        self.ls.append(new_l)
        # Store the new polyak average
        self.avg_ls.append(new_avg)
        # Update scale. Use the averaged one!
        self.s = np.exp(new_avg)

    def project_original(self, x, v, Q):
        """Finds a such that q(x + v + a*Q) = 0"""
        out = root(lambda a: self.manifold.q(x + v + Q @ a), np.array([self.a_guess]), tol=self.tol, options={'maxfev':self.maxiter})
        return (out.x, out.success, out.nfev , 0, 0) # output flag for accept/reject
    
    def acceptance_rate(self):
        """Computes total acceptance rate."""
        return sum(np.array(self.events) == 'Success') / len(self.events)

    def projection_failed_rate(self):
        """Computes rate of failed first projection"""
        return sum(np.array(self.events) == 'ProjFailed') / len(self.events)

    def mh_rejection_rate(self, over_proj_succeded=False):
        """Computes rate of failed MH steps."""
        if over_proj_succeded:
            return sum(np.array(self.events) == 'MHRejection') / sum(np.array(self.events) != 'ProjFailed')
        else:
            return sum(np.array(self.events) == 'MHRejection') / len(self.events)

    def reproj_failed_rate(self, over_mh_accepted=True):
        """Reprojection failure rate, when considered after MH accepted."""
        if over_mh_accepted:
            return sum(np.array(self.events) == "ReprojFailed") / sum(np.array(self.events) != "MHRejection")
        else:
            return sum(np.array(self.events) == "ReprojFailed") / len(self.events)



        