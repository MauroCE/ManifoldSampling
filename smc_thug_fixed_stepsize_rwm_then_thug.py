"""
Same as smc_thug_fixed_stepsize.py but we start off with RWM as the mutation kernel,
and once \epsilon is small enough (i.e. RWM fails), we switch to adaptive THUG.
"""
import numpy as np
from numpy import arange, ones, array, zeros, concatenate, hstack, unique, mean
from numpy import quantile, cov, eye, log, ceil, exp, clip, errstate, vstack
from numpy import array_equal
from numpy.linalg import cholesky, norm
from numpy.random import choice, uniform
from scipy.stats import multivariate_normal as MVN
from time import time

from tangential_hug_functions import HugTangentialStepEJSD
from tangential_hug_functions import HugTangentialPCStep
from tangential_hug_functions import HugTangential, HugTangentialPC
from tangential_hug_functions import Hug, HugPC

from RWM import RWM, RWM_Cov
from warnings import catch_warnings, filterwarnings


class SMCTHUG:
    def __init__(self, N, d, ystar, logprior, ϵmin=None, pmin=0.2, pter=0.01, tolscheme='unique', η=0.9, mcmc_iter=5, iterscheme='fixed', propPmoved=0.99, δ0=0.2, minstep=0.1, maxstep=100.0, a_star=0.3, B=5, manual_initialization=False, maxiter=300, maxMCMC=10, precondition=False, force_hug=False, αmax=0.999, αmin=0.01, pter_multiplier=1.1):
        """SMC sampler using Hug/Thug kernel. However we always start with a
        RWM kernel and then switch to THUG (with or without adaptive step size)
        once RWM starts failing. For this reason, we don't need the `thug` flag.

        N     : Number of particles
        d     : dimensionality of each particle
        ystar : Observed data for ABC
        logprior : Evaluates log prior at ξ
        ϵmin  : minimum tolerance. When we reach a smaller tolerance, we stop.
        pmin  : minimum acceptance prob we aim for. This is used to tune step size.
        pter  : terminal acceptance prob. When we go below this, then we stop. Used in stopping criterion.
        tolscheme : Either 'unique' or 'ess'. Determines how next ϵ is chosen.
        η     : quantile to use when determining a tolerance.
        mcmc_iter : Initial number of MCMC iterations for each particle.
        iterscheme : Whether to keep a fixed number of mcmc_iter per particle
                     or to find it adaptively using estimate acceptance probability.
                     Choose between 'fixed' and 'adaptive'.
        δ0      : Initial step size for THUG kernel.
        minstep : minimum step size for adaptive step size finding.
        maxstep : maximum stepsize for adaptive step size finding.
        B : Number of bounces in Thug
        manual_initialization : If true then user can set self.initialize_particles
                                to a custom function instead of initializing from
                                the prior.
        maxiter: Maximum number of SMC iterations. Used in self.stopping_criterion
        maxMCMC: Maximum number of MCMC steps. Used when iterscheme='adaptive'
        precondition: Boolean. Whether at each step we use ThugPC or Thug.
        force_hug: If true, then we use hug with alpha=0.0. That is we don't use thug.
        """
        # Store variables
        self.d = d                      # Dimensionality of each particle
        self.ystar = ystar              # Observed data
        self.ϵmin = ϵmin                # Lowest tolerance (for stopping)
        self.pmin = pmin                # Minimum acceptance prob (for stopping)
        self.pter = pter
        self.t = 0                      # Initialize iteration to 0
        self.η = η                      # Quantile for ϵ scheme
        self.a_star = a_star            # Target acceptance probability
        self.pPmoved = propPmoved       # Proportion of particles moved
        self.αmin = αmin
        self.α = 0.0 if force_hug else self.αmin  # Initial squeezing parameter
        self.B = B
        self.q = MVN(zeros(self.d), eye(self.d))
        self.mcmc_iter = mcmc_iter
        self.N = N
        self.minstep = minstep
        self.maxstep = maxstep
        self.manual_initialization = manual_initialization
        self.maxiter = maxiter
        self.total_time = 0.0
        self.maxMCMC = maxMCMC
        self.precondition = precondition
        self.force_hug = force_hug
        self.δ0 = δ0   # NEW LINE
        self.αmax = αmax
        self.initial_rwm_has_failed = False
        self.pter_multiplier = pter_multiplier  # Used to figure out when THUG should kick in
        assert pter_multiplier >= 1.0, "pter multiplier must be larger than 1.0."

        # Initialize arrays
        self.W       = zeros((N, 1))           # Normalized weights
        self.D       = zeros((N, 1))           # Distances
        self.A       = zeros((N, 1), dtype=int)           # Ancestors
        self.P       = zeros((N, self.d, 1))   # Particles
        self.EPSILON = [np.inf]                # ϵ for all iterations
        self.ESS     = [0.0]                   # ESS
        self.n_unique_particles = [0.0]
        self.n_unique_starting = []
        self.avg_acc_prob_within_MCMC = zeros((N, 1)) # (n, t) entry is the average acceptance probability of self.MCMC[self.t] iterations stpes of MCMC
        #self.accepted = zeros((N, 1))          # proportion of accepted thug steps within self.THUG
        self.MCMC_iter = [mcmc_iter]           # MCMC iterations per particle (K in Chang's code)
        self.accprob = [1.0]                   # Current acceptance probability
        self.step_sizes = [self.δ0]            # Tracks step sizes
        self.ALPHAS = [self.α]                 # Tracks the α for THUG

        # Store log prior
        self.logprior = logprior
        self.Σ = eye(self.d)
        self.Σfunc = lambda x: self.Σ

        # We only have one stopping criterion: ϵmin, pter and RWM/THUG
        if (ϵmin is None) or (pter is None):
            raise NotImplementedError("Arguments ϵmin and pter mustn't be None.")
        else:
            self.stopping_criterion = self.stopping_criterion_rwm
            print("### Stopping Criterion: Minimum Tolerance {} and Terminal Acceptance Probability {}".format(ϵmin, pter))

        # Set tolerance scheme
        if tolscheme == 'unique':
            self.tol_scheme = self.unique_tol_scheme
        elif tolscheme == 'ess':
            self.tol_scheme = self.ess_tol_scheme
        else:
            raise NotImplementedError("Tolerance schemes: unique or ess.")

        # Set iteration scheme
        if iterscheme == 'fixed':
            self.compute_n_mcmc_iterations = self.fixed_n_mcmc
        elif iterscheme == 'adaptive':
            self.compute_n_mcmc_iterations = self.adaptive_n_mcmc
        else:
            raise NotImplementedError("You must set `iterscheme` to either `fixed` or `adaptive`.")

        wrapMCMCoutput = lambda samples, acceptances: (samples[-1, :], mean(acceptances))
        # Set kernel to RWM (either preconditioned or isotropic)
        if precondition:
            print("### MCMC kernel: RWM with Preconditioning.")
            self.MCMCkernel = lambda *args: wrapMCMCoutput(*RWM_Cov(*args))
            self.MCMC_args  = lambda x0, N: (x0, self.Σ, N, self.logpi)
            self.estimateΣ  = lambda: cov(self.P[self.W[:, -2] > 0, :, -2].T) + 1e-8*eye(self.d)
        else:
            print("### MCMC kernel: isotropic RWM.")
            self.MCMCkernel = lambda *args: wrapMCMCoutput(*RWM(*args))
            self.MCMC_args  = lambda x0, N: (x0, self.B*self.step_sizes[-1], N, self.logpi)
            self.estimateΣ  = lambda: eye(self.d)
        # Set up a kernel containing the correct THUG kernel (Adaptive, non adaptive, preconditioned or not).
        # the MCMC kernel will be assigned to this when RWM fails.
        if not force_hug:   ##### THUG
            if precondition:
                print("### THUG kernel (for later): THUG with Preconditioning.")
                self.THUGkernel = lambda *args: wrapMCMCoutput(*(HugTangentialPC(*args)))
                self.THUG_args  = lambda x0, N: (x0, self.B * self.step_sizes[-1], self.B, self.Σfunc, N, self.α, self.q, self.logpi, self.grad_h)
                self.THUGestimateΣ  = lambda: cov(self.P[self.W[:, -2] > 0, :, -2].T) + 1e-8*eye(self.d)
            else:
                print("### THUG kernel (for later): THUG.")
                self.THUGkernel = lambda *args: wrapMCMCoutput(*(HugTangential(*args)))
                self.THUG_args  = lambda x0, N: (x0, self.B * self.step_sizes[-1], self.B, N, self.α, self.q, self.logpi, self.grad_h)
                self.THUGestimateΣ  = lambda: eye(self.d)
        else:   #### HUG
            if precondition:
                print("### THUG kernel (for later): HUG with Preconditioning.")
                self.THUGkernel = lambda *args: wrapMCMCoutput(*(HugPC(*args)))
                self.THUG_args  = lambda x0, N: (x0, self.B * self.step_sizes[-1], self.B, self.Σfunc, N, self.q, self.logpi, self.grad_h)
                self.THUGestimateΣ  = lambda: cov(self.P[self.W[:, -2] > 0, :, -2].T) + 1e-8*eye(self.d)
            else:
                print("### THUG kernel (for later): HUG.")
                self.THUGkernel = lambda *args: wrapMCMCoutput(*(Hug(*args)))
                self.THUG_args  = lambda x0, N: (x0, self.B * self.step_sizes[-1], self.B, N, self.q, self.logpi, self.grad_h)
                self.THUGestimateΣ  = lambda: eye(self.d)

        ### Finally, if using HUG or RWM simply remove the α update
        if force_hug:
            self.update_α = lambda a_hat, i: None

    @staticmethod
    def sample_prior():
        """Samples xi = (theta, z) from prior distribution."""
        raise NotImplementedError

    # def min_tol_or_min_acc_prob(self): return (self.EPSILON[-1] > self.ϵmin) and (self.t < self.maxiter) and (self.accprob[-1] > self.pter)
    def stopping_criterion_rwm(self): return (self.t < self.maxiter) and (self.EPSILON[-1] > self.ϵmin) # Only check maxiter and epsilon
    def stopping_criterion_thug(self): return (self.t < self.maxiter) and (self.EPSILON[-1] > self.ϵmin) and (self.accprob[-1] > self.pter) # Additionally, check for pter

    def unique_tol_scheme(self): return max(self.ϵmin, quantile(unique(self.D[self.A[:, -1], -1]), self.η))
    def ess_tol_scheme(self):    return max(self.ϵmin, quantile(self.D[self.A[:, -1], -1], self.η))

    def fixed_n_mcmc(self):    return self.mcmc_iter
    def adaptive_n_mcmc(self): return min(self.maxMCMC, int(ceil(log(1 - self.pPmoved) / log(1 - self.accprob[-1]))))

    @staticmethod
    def h(ξ, ystar):
        """Computes ||f(xi) - y*||"""
        raise NotImplementedError

    @staticmethod
    def h_broadcast(ξ_matrix, ystar):
        """Computes ||f_broadcast(xi) - y*||"""
        raise NotImplementedError

    def logkernel(self, ξ):
        """Kernel used for logpi. Epanechnikov in this case."""
        u = self.h(ξ, self.ystar)
        ϵ = self.EPSILON[self.t]
        with errstate(divide='ignore'):
            return log((3*(1 - (u**2 / (ϵ**2))) / (4*ϵ)) * float(u <= ϵ))

    def logpi(self, ξ):
        """Target distribution."""
        return self.logprior(ξ) + self.logkernel(ξ)

    @staticmethod
    def grad_h(ξ):
        """Computes the gradient of h(xi). Used by HUG/THUG."""
        raise NotImplementedError

    def compute_distances(self, flag=None):
        """Computes distance between all particles and ystar. If `flag` is
        provided, then it only computes the distance of the particles
        whose flag is True."""
        if flag is None:
            return self.h_broadcast(self.P[:, :, -1], self.ystar)
        else:
            return self.h_broadcast(self.P[flag, :, -1], self.ystar)

    def compute_distance(self, ix):
        """Computes distance between ix particle and ystar."""
        return self.h(self.P[ix, :, -1], self.ystar)

    @staticmethod
    def get_γ(i):
        """User needs to set this method. Returns the step size for the α update."""
        raise NotImplementedError

    def update_α(self, a_hat, i):
        """Updates α based on current acceptance probability"""
        τ = log(self.α / (1 - self.α))
        γ = self.get_γ(i)
        τ = τ - γ*(a_hat - self.a_star)
        self.α = np.clip(1 / (1 + exp(-τ)), self.αmin , self.αmax)

    def resample(self):
        """Resamples indeces of particles"""
        return choice(arange(self.N), size=self.N, replace=True, p=self.W[:, -1])

    @staticmethod
    def initialize_particles(N):
        """Can be used to initialize particles in a different way"""
        raise NotImplementedError("If manual_initialization=True then you must provide initialize_particles.")

    def sample(self):
        initial_time = time()
        # Initialize particles either manually
        if self.manual_initialization:
            particles = self.initialize_particles(self.N)
            for i in range(self.N):
                self.P[i, :, 0] = particles[i, :]
                self.W[i, 0]    = 1 / self.N
            print("### Particles have been initialized manually.")
        else:
            # or automatically from the prior
            for i in range(self.N):
                self.P[i, :, 0] = self.sample_prior()  # Sample particles from prior
                self.W[i, 0]    = 1 / self.N           # Assign uniform weights
            print("### Particles have been initialized from the prior.")

        # Compute distances. Use largest distance as current ϵ
        self.D[:, 0]    = self.compute_distances() # Compute distances
        self.EPSILON[0] = np.max(self.D[:, 0])     # Reset ϵ0 to max distance
        self.ESS[0]     = 1 / (self.W[:, 0]**2).sum()
        self.n_unique_particles[0] = len(unique(self.D[:, 0]))
        print("### Starting with {} unique particles.".format(self.n_unique_particles[0]))

        # Run Algorithm until stopping criteria is met
        print(self.t, self.maxiter, self.EPSILON[-1], self.ϵmin)
        print(self.stopping_criterion_rwm())
        print(self.t < self.maxiter, self.EPSILON[-1] > self.ϵmin)
        while self.stopping_criterion():
            # RESAMPLING
            self.A[:, self.t] = self.resample()
            self.t += 1

            # SELECT TOLERANCE
            self.EPSILON.append(self.tol_scheme())

            # ADD ZERO-COLUMN TO ALL MATRICES FOR STORAGE OF THIS ITERATION
            self.A = hstack((self.A, zeros((self.N, 1), dtype=int)))
            self.D = hstack((self.D, zeros((self.N, 1))))
            self.W = hstack((self.W, zeros((self.N, 1))))
            self.P = concatenate((self.P, zeros((self.N, self.d, 1))), axis=2)
            self.avg_acc_prob_within_MCMC = hstack((self.avg_acc_prob_within_MCMC, zeros((self.N, 1))))

            # COMPUTE WEIGHTS
            self.W[:, -1] = self.D[self.A[:, -2], -2] < self.EPSILON[-1]
            # Normalize weights (careful, some might be zeros or NaNs.)
            with catch_warnings():
                filterwarnings('error')
                try:
                    self.W[:, -1] = self.W[:, -1] / self.W[:, -1].sum()  # Normalize
                except RuntimeWarning:
                    print("There's some issue with the weights. Exiting.")
                    return {
                        'P': self.P,
                        'W': self.W,
                        'A': self.A,
                        'D': self.D,
                        'EPSILON': self.EPSILON,
                        'AP': self.accprob,
                        'MCMC_ITERS': self.MCMC_iter[:-1],
                        'STEP_SIZES': self.step_sizes[:-1],
                        'ESS': self.ESS,
                        'UNIQUE_PARTICLES': self.n_unique_particles,
                        'UNIQUE_STARTING': self.n_unique_starting,
                        'ALPHAS': self.ALPHAS,
                        'TIME': self.total_time
                    }

            # COMPUTE ESS
            self.ESS.append(1 / (self.W[:, -1]**2).sum())

            print("\n### SMC step: ", self.t)
            self.n_unique_starting.append(len(unique(self.D[self.A[:, -2], -2])))  # Unique after resampling
            print("ϵ = {:.10f}\t N unique starting: {}".format(round(self.EPSILON[-1], 5), self.n_unique_starting[-1]))

            # COMPUTE COVARIANCE MATRIX (d x d) from previous weights
            self.Σ = self.estimateΣ()

            # METROPOLIS-HASTINGS - MOVE ALIVE PARTICLES
            print("Metropolis-Hastings steps: ", self.MCMC_iter[-1])
            alive = self.W[:, -1] > 0.0     # Boolean flag for alive particles
            index = np.where(alive)[0]      # Indices for alive particles
            for ix in index:
                self.P[ix, :, -1], self.avg_acc_prob_within_MCMC[ix, -1] = self.MCMCkernel(*self.MCMC_args(self.P[self.A[ix, -2], :, -2], self.MCMC_iter[-1]))
                self.D[ix, -1] = self.compute_distance(ix)
            self.n_unique_particles.append(len(unique(self.D[alive, -1])))

            # ESTIMATE ACCEPTANCE PROBABILITY
            self.accprob.append(self.avg_acc_prob_within_MCMC[:, -1].mean())
            try:
                self.MCMC_iter.append(self.compute_n_mcmc_iterations())
                print("Average Acceptance Probability: {:.4f}".format(self.accprob[-1]))
            except OverflowError:
                print("Failed to compute n_mcmc_iterations. Current accept probability is likely 0.0. Exiting. ")
                break

            # DO NOT TUNE STEP SIZE
            self.step_sizes.append(self.δ0)
            print("Stepsize used in next SMC iteration: {:.4f}".format(self.step_sizes[-1]))

            # TUNE SQUEEZING PARAMETER FOR THUG ONLY IF RWM HAS ALREADY FAILED
            if self.initial_rwm_has_failed:
                self.update_α(self.accprob[-1], self.t)
                self.ALPHAS.append(self.α)
                print("Alpha used in next SMC iteration: {:.4f}".format(self.α))

            if self.EPSILON[-1] == self.ϵmin:
                print("Latest ϵ == ϵmin. Breaking")
                break

            # IF RWM IS STILL GOING, CHECK THE STOPPING CRITERION. IF IT FAILS
            # (I.E. RWM IS NOW TRYING TO SAMPLE FROM A FILAMENTARY DISTRIBUTION)
            # THEN CHANGE THE STOPPING CRITERION AND SWITCH TO USING THUG.
            if not self.initial_rwm_has_failed:
                # Check if pter has been reached. (actually, start it just before,
                # that's why it's multiplied by 1.1)
                if self.accprob[-1] <= (self.pter*self.pter_multiplier):
                    print("##############################################")
                    print("########### Initial RWM has failed ###########")
                    print("##############################################")
                    # RWM has failed
                    self.initial_rwm_has_failed = True
                    # Change stopping criterion
                    self.stopping_criterion = self.stopping_criterion_thug
                    # Change transition kernel to Thug
                    self.MCMCkernel = self.THUGkernel
                    self.MCMC_args  = self.THUG_args
                    self.estimateΣ  = self.THUGestimateΣ
                    # Store the iteration number at which we switch
                    self.t_at_which_we_switched_to_thug = self.t


        self.total_time = time() - initial_time

        return {
            'P': self.P,
            'W': self.W,
            'A': self.A,
            'D': self.D,
            'EPSILON': self.EPSILON,
            'AP': self.accprob,
            'MCMC_ITERS': self.MCMC_iter[:-1],
            'STEP_SIZES': self.step_sizes[:-1],
            'ESS': self.ESS,
            'UNIQUE_PARTICLES': self.n_unique_particles,
            'UNIQUE_STARTING': self.n_unique_starting,
            'ALPHAS': self.ALPHAS,
            'TIME': self.total_time,
            'SWITCH_TO_THUG': self.t_at_which_we_switched_to_thug
        }



def computational_cost(smc_output):
    T = len(smc_output['EPSILON']) - 1
    cost = 0
    for n in range(T):
        cost += np.sum(smc_output['W'][:, n+1] > 0) * smc_output['MCMC_ITERS'][n]
    number_of_produced_samples = np.sum(smc_output['W'][:, -1] > 0)
    return cost / number_of_produced_samples