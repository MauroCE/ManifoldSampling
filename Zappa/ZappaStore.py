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


class Zappa:

    def __init__(self, x0, manifold, logf, logp, n, sigma, tol, a_guess, maxiter=50, root=None):
        """
        Zappa sampling but stores everything.
        """
        # Choose root-finding function
        self.project = self.project_original
        if root == 'root_jacobian':
            self.project = self.project_root
        elif root == 'newton':
            self.project = self.project_newton

        # Store variables
        self.x0 = x0
        self.manifold = manifold
        self.logf = logf
        self.logp = logp
        self.n = int(n)
        self.sigma = sigma
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
        self.log_ap = None
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

        # Reset all of them to None, except x
        self.y = self.vx = self.vy  = self.nfevx = self.nfevy = self.njevx = self.njevy = None
        self.flagx = self.flagy = self.ax = self.ay = self.statusx = self.statusy = self.logap = None
        self.gx_basis = self.gy_basis = self.tx_basis = self.ty_basis = np.array([None, None])
        self.v_sample = self.v_prime_sample = None
        self.Qx = self.Qy = np.array([None, None])

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
            'Qys': self.Qys
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
            self.v_sample = self.sigma*randn(self.d)                # Isotropic MVN with scaling sigma         Size: (d, )
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

    def project_original(self, x, v, Q):
        """Finds a such that q(x + v + a*Q) = 0"""
        out = root(lambda a: self.manifold.q(x + v + Q @ a), np.array([self.a_guess]), tol=self.tol, options={'maxfev':self.maxiter})
        return (out.x, out.success, out.nfev , 0, 0) # output flag for accept/reject

    def project_root(self, x, v, Q):
        """Uses scipy.optimize.root with Jacobian."""
        out = _root_hybr(
            lambda a: self.manifold.q(x + v + Q @ a), np.array([self.a_guess]), 
            jac=lambda a: self.manifold.Q(x + v + Q @ a).T @ Q, 
            col_deriv=True, xtol=self.tol, maxfev=self.maxiter
        )
        return out.x, out.success, out.nfev, out.njev, out.status

    def project_newton(self, x, v, Q):
        """Uses newton method"""
        a = np.array(self.a_guess)   # (m, ) One entry for each constraint
        flag = 1
        i = 0
        # Newton's algorithm
        while la.norm(self.manifold.q(x + v + Q @ a)) > self.tol:
            # Notice that I need to wrap -q(x + v + Q @ a) into a Numpy Array because it returns a scalar.
            # Otherwise la.solve() will complain. This will NOT work for m > 1 constraints. In that case,
            # I would need to rewrite the function q of the manifold to return an array of size (m,)
            delta_a = la.solve(self.manifold.Q(x + v + Q @ a).T @ Q, np.array([-self.manifold.q(x + v + Q @ a)]))
            a += delta_a
            i += 1
            if i > self.maxiter:
                flag = 0
                return a, flag, i, i, 0
        return a, flag, i, i, 0
    
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



        