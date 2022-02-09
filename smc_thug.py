import numpy as np
from numpy import arange, ones, array, zeros, concatenate, hstack, unique
from numpy import quantile, cov, eye, log, ceil, exp, clip
from numpy.linalg import cholesky
from numpy.random import choice, uniform

from tangential_hug_functions import HugStepEJSD_Deterministic


class SMCTHUG:
    def __init__(self, N, d, ystar, ϵmin=None, pmin=None, tolscheme='unique', η=0.9, ϵmax=1.0, mcmc_iter=5, iterscheme='fixed', propPmoved=0.99, initStep=0.2, minstep=0.1, maxstep=1.0):
        """SMC sampler using Hug/Thug kernel.
        N     : Number of particles
        d     : dimensionality of each particle
        ystar : Observed data for ABC
        ϵmin  : minimum tolerance. When we reach a smaller tolerance, we stop.
        pmin  : minimum acceptance prob. When we reach a smaller prob, we stop.
        tolscheme : Either 'unique' or 'ess'. Determines how next ϵ is chosen.
        η     : quantile to use when determining a tolerance.
        ϵmax  : maximum tolerance, used during tolscheme.
        mcmc_iter : Initial number of MCMC iterations for each particle.
        iterscheme : Whether to keep a fixed number of mcmc_iter per particle
                     or to find it adaptively using estimate acceptance probability.
                     Choose between 'fixed' and 'adaptive'.
        initStep : Initial step size for MCMC kernel.
        minstep : minimum step size for adaptive step size finding.
        maxstep : maximum stepsize for adaptive step size finding.
        """
        # Store variables
        self.d = d                      # Dimensionality of each particle
        self.ystar = ystar              # Observed data
        self.ϵmin = ϵmin                # Lowest tolerance (for stopping)
        self.ϵmax = ϵmax                # Max tol for tol schedule
        self.pmin = pmin                # Minimum acceptance prob (for stopping)
        self.t = 0                      # Initialize iteration to 0
        self.η = η                      # Quantile for ϵ scheme
        self.pPmoved = propPmoved       # Proportion of particles moved

        # Initialize arrays
        self.W       = zeros((N, 1))           # Normalized weights
        self.D       = zeros((N, 1))           # Distances
        self.A       = zeros((N, 1))           # Ancestors
        self.P       = zeros((N, self.d, 1))   # Particles
        self.EPSILON = [np.inf]                # ϵ for all iterations
        self.ESS     = [0.0]                   # ESS
        self.n_unique_particles = [0.0]
        self.n_unique_starting = []
        self.accepted = zeros(N)               # accepted MCMC steps (?)
        self.MCMC_iter = [mcmc_iter]           # MCMC iterations per particle (K in Chang's code)
        self.accprob = [1.0]                   # Current acceptance probability
        self.step_sizes = [initStep]           # Tracks step sizes


        # Set stopping criterion or raise an error
        if ϵmax is not None:
            self.stopping_criterion = self.min_tolerance
        elif pmin is not None:
            self.stopping_criterion = self.min_acc_prob
        else:
            raise NotImplementedError("You must set one of `ϵmax` or `pmin`.")

        # Set tolerance scheme
        if tolscheme == 'unique':
            self.tol_scheme = self.unique_tol_scheme
        elif tolscheme == 'ess':
            self.tol_scheme = self.ess_tol_scheme
        else:
            raise NotImplementedError("Tolerance schemes: unique or ess.")

        # Set iteration scheme
        if iterscheme == 'fixed':
            self.compute_n_mcmc_iterations = self.fixed_n_mcmc()
        elif iterscheme == 'adaptive':
            self.compute_n_mcmc_iterations = self.adaptive_n_mcmc()
        else:
            raise NotImplementedError("You must set `iterscheme` to either `fixed` or `adaptive`.")

    def sample_prior(self):
        """Samples xi = (theta, z) from prior distribution.
        This is the prior of G and K model.""".
        theta = uniform(low=0.0, high=10.0, size=4)
        z = randn(self.d - 4)
        return concatenate((theta, z))

    def min_tolerance(self): return self.EPSILON[-1] > self.ϵmin
    def min_acc_prob(self):  return self.accprob[-1] > self.pmin

    def unique_tol_scheme(self): return max(self.ϵmax, quantile(unique(self.D[self.A[:, -1], -1]), self.η))
    def ess_tol_scheme(self):    return max(self.ϵmax, quantile(self.D[self.A[:, -1], -1], self.η))

    def fixed_n_mcmc(self):    return self.mcmc
    def adaptive_n_mcmc(self): return int(ceil(log(1 - self.pPmoved) / log(1 - self.accprob[-1])))

    @staticmethod
    def h(ξ_matrix, ystar):
        """Computes ||f_broadcast(xi) - y*||"""
        pass

    @staticmethod
    def grad_h(ξ):
        """Computes the gradient of h(xi). Used by HUG/THUG."""
        pass

    def compute_distances(self, flag=None):
        """Computes distance between all particles and ystar. If `flag` is
        provided, then it only computes the distance of the particles
        whose flag is True."""
        if ix is None:
            return self.h(self.P[:, :, -1], self.ystar)
        else:
            return self.h(self.P[flag, :, -1], self.ystar)

    def resample(self):
        """Resamples indeces of particles"""
        return choice(arange(self.N), size=N, replace=True, p=self.W[:, -1])

    def THUG():
        """HUG/THUG kernel."""
        HugStepEJSD_Deterministic(x0, v0, logu, T, B, q, logpi, self.grad_h)
        pass

    def sample(self):
        # Initialize particles
        for i in range(self.N):
            self.P[i, :, 0] = self.sample_prior()  # Sample particles from prior
            self.W[i, 0]    = 1 / self.N           # Assign uniform weights

        # Compute distances. Use largest distance as current ϵ
        self.D[:, 0]    = self.compute_distances() # Compute distances
        self.EPSILON[0] = np.max(self.D[:, 0])     # Reset ϵ0 to max distance
        self.ESS[0]     = 1 / (self.W[:, 0]**2).sum()
        self.n_unique_particles[0] = len(unique(self.D[:, 0]))

        # Run Algorithm until stopping criteria is met
        while self.stopping_criterion():
            # RESAMPLING
            self.A[:, t] = self.resample()
            self.t += 1

            # SELECT TOLERANCE
            self.EPSILON.append(self.tol_scheme())

            # ADD ZERO-COLUMN TO ALL MATRICES FOR STORAGE OF THIS ITERATION
            self.A = hstack(self.A, zeros((self.N, 1)))
            self.D = hstack(self.D, zeros((self.N, 1)))
            self.W = hstack(self.W, zeros((self.N, 1)))
            self.P = concatenate((self.P, zeros((self.N, self.d))), axis=2)

            # COMPUTE WEIGHTS
            self.W[:, -1] = self.D[self.A[:, -2], -2] < self.EPSILON[-1]
            self.W[:, -1] = self.W[:, -1] / self.W[:, -1].sum()  # Normalize

            # COMPUTE ESS
            self.ESS.append(1 / (self.W[:, -1]**2).sum())

            print("SMC step: ", self.t)
            self.n_unique_starting.append(len(unique(self.D[self.A[:, -2], -2])))
            print("ϵ = ", round(self.EPSILON[-1], 5), " N unique starting: ", self.n_unique_starting[-1])

            # COMPUTE COVARIANCE MATRIX (d x d) from previous weights
            Σ = cov(self.P[self.W[:, -2] > 0, :, -2].T) + 1e-8*eye(self.d)
            A = cholesky(Σ)

            # METROPOLIS-HASTINGS - MOVE ALIVE PARTICLES
            print("Metropolis-Hastings step.")
            alive = self.W[:, -1] > 0.0     # Boolean flag for alive particles
            index = np.where(alive)[0]      # Indices for alive particles
            for ix in index:
                self.P[ix, :, -1], self.accepted[ix] = self.THUG()
                self.D[ix, -1]    = self.compute_distances(flag=alive)
            self.n_unique_particles.append(len(unique(self.D[alive, -1])))

            # ESTIMATE ACCEPTANCE PROBABILITY
            self.accprob.append(self.accepted[alive].mean() / self.MCMC_iter[-1])
            self.MCMC_iter.append(self.compute_n_mcmc_iterations())
            print("Average Acceptance Probability: ", self.accprob[-1])

            # TUNE STEP SIZE
            self.step_sizes.append(clip(exp(log(self.step_sizes[-1]) + 0.5*(self.accprob[-1] - self.pmin)), minstep, maxstep))
            print("Stepsize used in next SMC iteration: ", self.step_sizes[-1])

            if self.EPSILON[-1] == self.ϵmax: break

        return {
            'P': self.P,
            'W': self.W,
            'A': self.A,
            'D': self.D,
            'EPSILON': self.EPSILON,
            'AP': self.accprob,
            'MCMC_ITERS': self.MCMC_iter[:-1],
            'STEP_SIZES': self.step_sizes,
            'ESS': self.ESS,
            'UNIQUE_PARTICLES': self.n_unique_particles,
            'UINQUE_STARTING': self.n_unique_starting
        }
