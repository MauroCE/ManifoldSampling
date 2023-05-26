"""
Various functions and classes for Markov Snippets.
"""
import numpy as np
from numpy import zeros, hstack, arange, repeat, log, concatenate, eye, full
from numpy import apply_along_axis, exp, unravel_index, dstack, vstack, ndarray
from numpy import zeros_like, quantile, unique, clip, where
from numpy.linalg import solve, norm
from numpy.random import rand, choice, normal, randn, randint, default_rng
from scipy.stats import multivariate_normal as MVN
from scipy.special import logsumexp
from time import time
from copy import deepcopy

from RWM import RWM, generate_RWMIntegrator
# from tangential_hug_functions import HugTangentialMultivariate
# from tangential_hug_functions import HugTangential
from Manifolds.Manifold import Manifold
from tangential_hug_functions import TangentialHugSampler
from Manifolds.GKManifoldNew import find_point_on_manifold

### Markov Snippets classes (Multivariate, i.e. use a Jacobian, not a gradient)


##### SINGLE MARKOV SNIPPETS CLASS: MULTI AND UNI VARIATE, ADAPTS BOTH TOLERANCES AND DELTA
class MSAdaptive:

    def __init__(self, SETTINGS):
        """Markov Snippets class that works for both univariate and multivariate
        problems. By univariate we mean that the constraint function f maps Rn
        to R and therefore its Jacobian is actually just a gradient vector.

        All these settings are customizable:

            1. THUG vs RWM integrator
            2. Fixed ϵ schedule or adaptive ϵ schedule (ϵn is chosen as a quantile
            of the distribution of distances ||f(x) - y|| of the x-part of the
            particles from the manifold.)
            3. Fixed step size δ or adaptive δ, chosen to match a given step size.
            4. If started with RWM, one can switch to THUG integrator based on
            several conditions:
                1.
                2.
                3.
        """
        # Store variables
        self.N    = SETTINGS['N']          # Number of particles
        self.B    = SETTINGS['B']          # Number of integration steps
        self.δ    = SETTINGS['δ']          # Step-size for each integration step
        self.d    = SETTINGS['d']          # Dim of x-component of particle
        self.δmin = SETTINGS['δmin']       # δ ≧ δmin when adaptive
        self.δmax = SETTINGS['δmax']       # δ ≦ δmax when step size is adaptive
        self.εmin = SETTINGS['εmin']       # ε ≧ εmin when tolerance is adaptive
        self.min_pm = SETTINGS['min_pm'] # if fewer resampled particles have k ≧ 1 than self.min_pm we stop the algorithm (pm stands for prop-moved)
        self.maxiter = SETTINGS['maxiter'] # MS stopped when n ≧ maxiter
        self.verbose = SETTINGS['verbose'] # Whether to print MS progress
        self.εs_fixed = SETTINGS['εs_fixed']   # Sequence of tolerances, this is only used if adaptiveϵ is False
        self.manifold = SETTINGS['manifold'] # Manifold around which we sample
        self.ε0_manual = SETTINGS['ε0_manual'] # When initialization is 'manual' then this is the ϵ0 that we assume it has been sampled from
        self.adaptiveε = SETTINGS['adaptiveε'] # If true, we adapt ε based on the distances, otherwise we expect a fixed sequence
        self.adaptiveδ = SETTINGS['adaptiveδ'] # If true, we adapt δ based on the proxy acceptance probability.
        self.z0_manual = SETTINGS['z0_manual'] # Starting particles, used only if initialization is manual
        self.pm_target = SETTINGS['pm_target'] # Target 'proportion of moved' particles the we aim to while adapting δ
        self.pm_switch = SETTINGS['pm_switch'] # when the proportion of particles moved is less than this, we switch
        self.prior_seed = SETTINGS['prior_seed'] # Seed to used when initializing the particles from the prior, to allow reproducibility
        self.low_memory = SETTINGS['low_memory'] # whether to use a low-memory version or not. Low memory does not store ZNK
        self.integrator = SETTINGS['integrator'] # Determines if we use RWM, THUG or start with RWM and switch to THUG
        self.εprop_switch = SETTINGS['εprop_switch'] # When (ε_n - ε_{n-1}) / ε_n is less than self.εprop_switch, then we switch
        self.quantile_value = SETTINGS['quantile_value'] # Used to determine the next ϵ
        self.initialization = SETTINGS['initialization'] # type of initialization to use
        self.switch_strategy = SETTINGS['switch_strategy'] # strategy used to determine when to switch RWM->THUG.

        # Check arguments types
        assert isinstance(self.N,  int), "N must be an integer."
        assert isinstance(self.B, int), "B must be an integer."
        assert isinstance(self.δ, float), "δ must be a float."
        assert isinstance(self.d, int), "d must be an integer."
        assert isinstance(self.δmin, float), "δmin must be float."
        assert isinstance(self.δmax, float), "δmax must be float."
        assert isinstance(self.εmin, float), "εmin must be float."
        assert isinstance(self.min_pm, float), "min_pm must be float."
        assert isinstance(self.maxiter, int), "maxiter must be integer."
        assert isinstance(self.verbose, bool), "verbose must be boolean."
        assert isinstance(self.εs_fixed, np.ndarray) or (self.εs_fixed is None), "εs must be a numpy array or must be None."
        assert isinstance(self.manifold, Manifold), "manifold must be an instance of class Manifold."
        assert isinstance(self.adaptiveε, bool), "adaptiveϵ must be bool."
        assert isinstance(self.adaptiveδ, bool), "adaptiveδ must be bool."
        assert isinstance(self.z0_manual, np.ndarray) or (self.z0_manual is None), "z0_manual must be a numpy array or None."
        assert isinstance(self.pm_target, float), "pm_target must be float."
        assert isinstance(self.pm_switch, float), "pm_switch must be float."
        assert isinstance(self.prior_seed, int), "prior_seed must be integer."
        assert isinstance(self.low_memory, bool), "low_memory must be bool."
        assert isinstance(self.integrator, str), "integrator must be a string."
        assert isinstance(self.εprop_switch, float), "εprop_switch must be a float."
        assert (self.ε0_manual is None) or isinstance(self.ε0_manual, float), "ε0_manual must be float or None."
        assert isinstance(self.quantile_value, float), "quantile_value must be a float."
        assert isinstance(self.initialization, str), "initialization must be a string."
        assert isinstance(self.switch_strategy, str), "switch_strategy must be a string."

        # Check argument values
        assert self.δ > 0.0, "δ must be larger than 0."
        assert (self.εs_fixed is None) or all(x>y for x, y in zip(self.εs_fixed, self.εs_fixed[1:])), "εs must be a strictly decreasing list, or None."
        assert self.δmin > 0.0, "δmin must be larger than 0."
        assert self.δmin <= self.δmax, "δmin must be less than or equal to δmax."
        assert self.εmin > 0.0, "εmin must be larger than 0."
        assert (self.min_pm >= 0.0) and (self.min_pm <= 1.0), "min_pm must be in [0, 1]."
        assert (self.pm_target >= 0) and (self.pm_target <= 1.0), "pm_target must be in [0, 1]."
        assert (self.pm_switch >= 0) and (self.pm_switch <= 1.0), "pm_switch must be in [0, 1]."
        assert self.integrator.lower() in ['rwm', 'thug', 'rwm_then_thug'], "integrator must be one of 'RWM', 'THUG', or 'RWM_THEN_THUG'."
        assert (self.εprop_switch >= 0.0) and (self.εprop_switch <= 1.0), "εprop_switch must be in [0, 1]."
        assert (self.ε0_manual is None) or (self.ε0_manual >= 0.0), "ε0_manual must be larger than 0 or must be None."
        assert (self.quantile_value >= 0) and (self.quantile_value <= 1.0), "quantile_value must be in [0, 1]."
        assert self.initialization in ['prior', 'manual'], "initialization must be one of 'prior' or 'manual'."
        assert self.switch_strategy in ['εprop', 'pm'], "switch_strategy must be one of 'εprop' or 'pm'."
        if isinstance(self.z0_manual, np.ndarray):
            if self.z0_manual.shape != (self.N, 2*self.d):
                raise ValueError("z0_manual must have shape (N, 2d).")

        # Create functions and variables based on input arguments
        self.verboseprint = print if self.verbose else lambda *a, **k: None  # Prints only when verbose is true
        self.univariate = True if (self.manifold.get_codimension() == 1) else False # basically keeps track if it is uni or multi variate.
        self.δs = [self.δ]
        self.switched = False

        # Choose correct integrator to use
        if (self.integrator.lower() == 'rwm') or (self.integrator.lower() == 'rwm_then_thug'):
            # Choose Random Walk Metropolis integrator
            self.verboseprint("Integrator: RWM.")
            self.ψ_generator = lambda B, δ: generate_RWMIntegrator(B, δ) # This is now a function that given B, δ it returns a function that integrates with those parameters
            self.ψ = self.ψ_generator(self.B, self.δ)
        elif self.integrator.lower() == 'thug':
            self.verboseprint("Integrator: THUG.")
            # Instantiate the class, doesn't matter which ξ0 or logpi we use.
            THUGSampler = TangentialHugSampler(self.manifold.sample(advanced=True), self.B*self.δ, self.B, self.N, 0.0, self.manifold.logprior, self.manifold.fullJacobian, method='linear', safe=True)
            self.ψ_generator = THUGsampler.generate_hug_integrator # again, this takes B, δ and returns an integrator (notice logpi doesn't matter)
            self.ψ = self.ψ_generator(self.B, self.δ)
        else:
            raise ValueError("Unexpected value found for integrator.")

        # When ϵs are fixed and given, one must have a manual initialization.
        if (not self.adaptiveε) and self.initialization == 'prior':
            raise NotImplementedError("Currently, one can only use a manual initialization when using a fixed schedule of ϵs.")

        # Choose correct initialization proceedure (and sort out self.ϵs and self.ηs)
        if self.initialization == 'prior' and (self.adaptiveε):
            # Initializion from the prior can only be done when we adapt ϵ
            self.εs = []  # Will be filled with self.ϵmax calculated from prior samples
            self.log_ηs = []  # Will be filled with a log_ηϵmax
            sample_positions = lambda: self.manifold.sample_prior(self.N, seed=self.prior_seed)
            sample_velocities = lambda: default_rng(seed=self.prior_seed).normal(size=(self.N, self.d))
            self.initializer = lambda: hstack((sample_positions(), sample_velocities()))
        elif (self.initialization == 'manual'):
            if (self.ε0_manual is not None) and self.adaptiveε:
                # One must provide ε0_manual
                self.εs = [self.ε0_manual]
                self.log_ηs = [FilamentaryDistribution(self.manifold.generate_logηε, self.ε0_manual)]
                self.initializer = lambda: self.z0_manual
            elif (self.ε0_manual is None) and (not self.adaptiveε):
                # both z0 and ϵs are provided
                self.εs = self.εs_fixed
                self.log_ηs = [FilamentaryDistribution(self.manifold.generate_logηε, ε) for ε in self.εs]
                self.initializer = lambda: self.z0_manual
            else:
                raise ValueError("Invalid initialization specifications.")
        else:
            raise ValueError("Invalid initialization specifications.")

        # Choose correct distance function
        if self.univariate:
            self.compute_distances = self._compute_distances_univariate
        else:
            self.compute_distances = self._compute_distances_multivariate

        # Determine whether to switch RWM -> THUG or not
        if self.integrator.lower() == 'rwm_then_thug':
            self.switch = True
        else:
            self.switch = False

    def _compute_nth_tolerance(self, z):
        """If the εs schedule is fixed, this does nothing. However, if the schedule
        is adaptive (i.e. self.adaptiveε == True), then we compute it as a quantile
        of distances of the particles from the manifold."""
        if not self.adaptiveε:
            pass
        else:
            # compute distances
            distances = self.compute_distances(z[:, :self.d])
            # add distances to storage
            self.DISTANCES = vstack((self.DISTANCES, distances))
            # determine next ε as quantile of distances
            ε = min(self.εs[self.n-1], quantile(unique(distances), self.quantile_value))
            self.εs.append(ε)
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηε, ε))

    def _compute_weights(self, log_μnm1_z, log_μn_ψk_z):
        """Computes weights using the log-sum-exp trick."""
        logw    = log_μn_ψk_z - log_μnm1_z  # unnormalized log-weights (N, B+1)
        logsumw = logsumexp(logw)           # total sum of weights (N, B+1)
        return exp(logw - logsumw)          # normalized weights (N, B+1)

    def _compute_distances_multivariate(self, x_particles):
        """Computes distances for a multivariate constraint function."""
        return norm(apply_along_axis(self.manifold.q, 1, x_particles), axis=1)

    def _compute_distances_univariate(self, x_particles):
        """Computes distances for a univariate constraint function."""
        return abs(apply_along_axis(self.manifold.q, 1, x_particles))

    def _update_δ_and_ψ(self):
        """If self.adaptiveδ is True, then we adapt based on proxy AP. Otherwise
        we keep it the same. When adapting δ, we must remember to adapt ψ since
        now the integrator will be different."""
        if self.adaptiveδ:
            self.δ = clip(exp(log(self.δ) + 0.5*(self.PROP_MOVED[self.n] - self.pm_target)), self.δmin, self.δmax)
            self.δs.append(self.δ)
            self.ψ = self.ψ_generator(self.B, self.δ)
            self.verboseprint("\tStep-size adapted to: {:.16f}".format(self.δ))
        else:
            self.δs.append(self.δ)
            self.verboseprint("\tStep-size kept fixed at: {:.16f}".format(self.δ))

    def switch_integrator(self):
        """Switches from RWM to THUG."""
        # the next 3 lines are taken verbatim from __init__ when integrator = 'THUG'
        x0 = self.manifold.sample(advanced=True)
        self.sampled_x0 = x0
        THUGSampler = TangentialHugSampler(x0, self.B*self.δ, self.B, self.N, 0.0, self.manifold.logprior, self.manifold.fullJacobian, method='linear', safe=True)
        self.ψ_generator = THUGSampler.generate_hug_integrator # again, this takes B, δ and returns an integrator (notice logpi doesn't matter)
        self.ψ = self.ψ_generator(self.B, self.δ)
        # Store when the switch happend
        self.n_switch = self.n  # store when the switch happens
        self.switched = True
        self.verboseprint("\n")
        self.verboseprint("####################################")
        self.verboseprint("### SWITCHING TO THUG INTEGRATOR ###")
        self.verboseprint("####################################")
        self.verboseprint("\n")

    def initialize_particles(self):
        """Initializes the particles and stores them in a separate variable, for checking purposes."""
        z0 = self.initializer()
        self.starting_particles = z0
        return z0

    def sample(self):
        """Samples using the Markov Snippets algorithm."""
        start_time = time()
        #### STORAGE
        self.ZN          = zeros((1, self.N, 2*self.d))            # z_n^{(i)}
        self.ZNK         = zeros((1, self.N*(self.B+1), 2*self.d)) # z_{n, k}^{(i)} all the N(T+1) particles
        self.Wbar        = zeros(self.N*(self.B+1))
        self.DISTANCES   = zeros(self.N)                      # distances are computed on the z_n^{(i)}
        self.ESS         = [self.N*(self.B+1)]                     # ESS computed on Wbar so in reference to all N(T+1) particles
        self.K_RESAMPLED = zeros(self.N)                    # Stores indeces resampled
        self.PROP_MOVED  = [1.0]                            # Stores proportion of particles moved forward on the trajectories
        #### INITIALIZATION
        z = self.initialize_particles()   # (N, 2d)
        self.ZN[0] = z
        if self.initialization == 'prior':
            distances = self.compute_distances(z[:, :self.d])  # compute εmax and log_ηεmax and add them to the storage lists
            self.εmax = np.max(distances)
            self.εs.append(self.εmax)
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηε, self.εmax))
            self.verboseprint("Setting initial epsilon to εmax = {:.16f}".format(self.εmax))
        # Keep running until stopping criterion is met
        # In this case we stop if we reach the number of maximum iterations, or
        # if out ε becomes smaller than εmin or if we move less than self.min_pm particles
        self.n = 1
        try:
            while (self.n <= self.maxiter) and (abs(self.εs[self.n-1]) >= self.εmin) and (self.PROP_MOVED[self.n-1] >= self.min_pm):
                self.verboseprint("Iteration: ", self.n)

                #### COMPUTE TRAJECTORIES
                Z = apply_along_axis(self.ψ, 1, z)                                        # (N, B+1, 2d)
                if not self.low_memory:
                    self.ZNK = vstack((self.ZNK, Z.reshape(1, self.N*(self.B+1), 2*self.d)))  # (n+1, N(B+1), 2d)
                self.verboseprint("\tTrajectories constructed.")

                #### DETERMINE TOLERANCE TO TARGET AT THIS ITERATION
                self._compute_nth_tolerance(z) # If adaptive, computes εn and logηεn, otherwise does nothing (already available when schedule is fixed)
                self.verboseprint("\tEpsilon: {:.16f}".format(self.εs[self.n]))

                #### COMPUTE WEIGHTS
                # Log-Denominator: shared for each point in the same trajectory
                log_μnm1_z  = apply_along_axis(self.log_ηs[self.n-1], 1, Z[:, 0, :self.d])         # (N, )
                log_μnm1_z  = repeat(log_μnm1_z, self.B+1, axis=0).reshape(self.N, self.B+1) # (N, B+1)
                # Log-Numerator: different for each point on a trajectory.
                log_μn_ψk_z = apply_along_axis(self.log_ηs[self.n], 2, Z[:, :, :self.d])         # (N, B+1)
                W = self._compute_weights(log_μnm1_z, log_μn_ψk_z)
                self.verboseprint("\tWeights computed and normalized.")
                # Store weights and ESS
                self.Wbar = vstack((self.Wbar, W.flatten()))
                self.ESS.append(1 / np.sum(W**2))

                #### RESAMPLING
                resampling_indeces = choice(a=arange(self.N*(self.B+1)), size=self.N, p=W.flatten())
                unravelled_indeces = unravel_index(resampling_indeces, (self.N, self.B+1))
                self.K_RESAMPLED = vstack((self.K_RESAMPLED, unravelled_indeces[1]))
                indeces = dstack(unravelled_indeces).squeeze()
                z = vstack([Z[tuple(ix)] for ix in indeces])     # (N, 2d)
                self.verboseprint("\tParticles Resampled.")

                #### REJUVENATE VELOCITIES
                z[:, self.d:] = normal(loc=0.0, scale=1.0, size=(self.N, self.d))
                self.ZN = vstack((self.ZN, z[None, ...]))
                self.verboseprint("\tVelocities refreshed.")

                #### ADAPT STEP SIZE
                # Compute proxy acceptance probability
                self.PROP_MOVED.append(sum(self.K_RESAMPLED[-1] >= 1) / self.N)
                self.verboseprint("\tProp Moved: {:.16f}".format(self.PROP_MOVED[self.n]))
                # Adapt δ basedn on proxy acceptance probability
                self._update_δ_and_ψ()

                #### CHECK IF IT'S TIME TO SWITCH INTEGRATOR
                if self.integrator.lower() == 'rwm_then_thug':  # only happens when we allow switching
                    if not self.switched:                       # continue ahead only if we haven't already switched
                        if self.switch_strategy == 'εprop':
                            if self.n >= 2:
                                if ((self.εs[self.n-1] - self.εs[self.n]) / self.εs[self.n-1]) <= self.εprop_switch:
                                    self.switch_integrator()
                        elif self.switch_strategy == 'pm':
                            if self.PROP_MOVED[self.n] <= self.pm_switch:
                                self.switch_integrator()
                        else:
                            raise ValueError("Invalid switching strategy.")
                self.n +=1
            self.total_time = time() - start_time
        except (ValueError, KeyboardInterrupt) as e:
            print("ValueError was raised: ", e)
        return z

##### SINGLE SMC CLASS: MULTI AND UNI VARIATE, ADAPTS BOTH TOLERANCES AND DELTA
class SMCAdaptive:

    def __init__(self, SETTINGS):
        """SMC class that can use RWM or THUG kernels, as well as switching to
        THUG at a later stage based on conditions. It's also possible to adapt
        ϵ and δ. Notice that we still use SETTINGS['integrator'] to determine
        which sampler to use. Altough the dictionary key is called 'integrator',
        here we actually use a stochastic kernel. In other words, RWM and THUG
        perform a Metropolis-Hastings step at the end of the trajectory.

        Remember: the key difference is that here only the position component is
        part of the particle. In Markov Snippets one uses z=[x, v] as particles,
        whereas here we use z=x."""
        # Store variables
        self.N    = SETTINGS['N']          # Number of particles
        self.B    = SETTINGS['B']          # Number of integration steps
        self.δ    = SETTINGS['δ']          # Step-size for each integration step
        self.d    = SETTINGS['d']          # Dim of x-component of particle
        self.δmin = SETTINGS['δmin']       # δ ≧ δmin when adaptive
        self.δmax = SETTINGS['δmax']       # δ ≦ δmax when step size is adaptive
        self.εmin = SETTINGS['εmin']       # ε ≧ εmin when tolerance is adaptive
        self.min_pm = SETTINGS['min_pm'] # if fewer resampled particles have k ≧ 1 than self.min_pm we stop the algorithm (pm stands for prop-moved)
        self.maxiter = SETTINGS['maxiter'] # MS stopped when n ≧ maxiter
        self.verbose = SETTINGS['verbose'] # Whether to print MS progress
        self.εs_fixed = SETTINGS['εs_fixed']   # Sequence of tolerances, this is only used if adaptiveϵ is False
        self.manifold = SETTINGS['manifold'] # Manifold around which we sample
        self.ε0_manual = SETTINGS['ε0_manual'] # When initialization is 'manual' then this is the ϵ0 that we assume it has been sampled from
        self.adaptiveε = SETTINGS['adaptiveε'] # If true, we adapt ε based on the distances, otherwise we expect a fixed sequence
        self.adaptiveδ = SETTINGS['adaptiveδ'] # If true, we adapt δ based on the proxy acceptance probability.
        self.z0_manual = SETTINGS['z0_manual'] # Starting particles, used only if initialization is manual
        self.pm_target = SETTINGS['pm_target'] # Target 'proportion of moved' particles the we aim to while adapting δ
        self.pm_switch = SETTINGS['pm_switch'] # when the proportion of particles moved is less than this, we switch
        self.prior_seed = SETTINGS['prior_seed'] # Seed to used when initializing the particles from the prior, to allow reproducibility
        self.low_memory = SETTINGS['low_memory'] # whether to use a low-memory version or not. Low memory does not store ZNK
        self.integrator = SETTINGS['integrator'] # Determines if we use RWM, THUG or start with RWM and switch to THUG
        self.εprop_switch = SETTINGS['εprop_switch'] # When (ε_n - ε_{n-1}) / ε_n is less than self.εprop_switch, then we switch
        self.quantile_value = SETTINGS['quantile_value'] # Used to determine the next ϵ
        self.initialization = SETTINGS['initialization'] # type of initialization to use
        self.switch_strategy = SETTINGS['switch_strategy'] # strategy used to determine when to switch RWM->THUG.

        # Check arguments types
        assert isinstance(self.N,  int), "N must be an integer."
        assert isinstance(self.B, int), "B must be an integer."
        assert isinstance(self.δ, float), "δ must be a float."
        assert isinstance(self.d, int), "d must be an integer."
        assert isinstance(self.δmin, float), "δmin must be float."
        assert isinstance(self.δmax, float), "δmax must be float."
        assert isinstance(self.εmin, float), "εmin must be float."
        assert isinstance(self.min_pm, float), "min_pm must be float."
        assert isinstance(self.maxiter, int), "maxiter must be integer."
        assert isinstance(self.verbose, bool), "verbose must be boolean."
        assert isinstance(self.εs_fixed, np.ndarray) or (self.εs_fixed is None), "εs must be a numpy array or must be None."
        assert isinstance(self.manifold, Manifold), "manifold must be an instance of class Manifold."
        assert isinstance(self.adaptiveε, bool), "adaptiveϵ must be bool."
        assert isinstance(self.adaptiveδ, bool), "adaptiveδ must be bool."
        assert isinstance(self.z0_manual, np.ndarray) or (self.z0_manual is None), "z0_manual must be a numpy array or None."
        assert isinstance(self.pm_target, float), "pm_target must be float."
        assert isinstance(self.pm_switch, float), "pm_switch must be float."
        assert isinstance(self.prior_seed, int), "prior_seed must be integer."
        assert isinstance(self.low_memory, bool), "low_memory must be bool."
        assert isinstance(self.integrator, str), "integrator must be a string."
        assert isinstance(self.εprop_switch, float), "εprop_switch must be a float."
        assert (self.ε0_manual is None) or isinstance(self.ε0_manual, float), "ε0_manual must be float or None."
        assert isinstance(self.quantile_value, float), "quantile_value must be a float."
        assert isinstance(self.initialization, str), "initialization must be a string."
        assert isinstance(self.switch_strategy, str), "switch_strategy must be a string."

        # Check argument values
        assert self.δ > 0.0, "δ must be larger than 0."
        assert (self.εs_fixed is None) or all(x>y for x, y in zip(self.εs_fixed, self.εs_fixed[1:])), "εs must be a strictly decreasing list, or None."
        assert self.δmin > 0.0, "δmin must be larger than 0."
        assert self.δmin <= self.δmax, "δmin must be less than or equal to δmax."
        assert self.εmin > 0.0, "εmin must be larger than 0."
        assert (self.min_pm >= 0.0) and (self.min_pm <= 1.0), "min_pm must be in [0, 1]."
        assert (self.pm_target >= 0) and (self.pm_target <= 1.0), "pm_target must be in [0, 1]."
        assert (self.pm_switch >= 0) and (self.pm_switch <= 1.0), "pm_switch must be in [0, 1]."
        assert self.integrator.lower() in ['rwm', 'thug', 'rwm_then_thug'], "integrator must be one of 'RWM', 'THUG', or 'RWM_THEN_THUG'."
        assert (self.εprop_switch >= 0.0) and (self.εprop_switch <= 1.0), "εprop_switch must be in [0, 1]."
        assert (self.ε0_manual is None) or (self.ε0_manual >= 0.0), "ε0_manual must be larger than 0 or must be None."
        assert (self.quantile_value >= 0) and (self.quantile_value <= 1.0), "quantile_value must be in [0, 1]."
        assert self.initialization in ['prior', 'manual'], "initialization must be one of 'prior' or 'manual'."
        assert self.switch_strategy in ['εprop', 'pm'], "switch_strategy must be one of 'εprop' or 'pm'."
        if isinstance(self.z0_manual, np.ndarray):
            if self.z0_manual.shape != (self.N, 2*self.d):
                raise ValueError("z0_manual must have shape (N, d).")
            # This is an SMC sampler, meaning the particles consist only of the position, not of the velocity.
            # Create a new variable, unique to SMC samplers, that stores the initial positions
            self.x0_manual = self.z0_manual[:, :self.d]


        # Create functions and variables based on input arguments
        self.verboseprint = print if self.verbose else lambda *a, **k: None  # Prints only when verbose is true
        self.univariate = True if (self.manifold.get_codimension() == 1) else False # basically keeps track if it is uni or multi variate.
        self.δs = [self.δ]
        self.switched = False

        # Choose correct KERNEL to use
        if (self.integrator.lower() == 'rwm') or (self.integrator.lower() == 'rwm_then_thug'):
            # Choose Random Walk Metropolis kernel
            self.verboseprint("Stochastic Kernel: RWM.")
            self.MH_kernel = lambda x, B, δ, log_ηε: RWM(x, B*δ, 1, log_ηε)[0].flatten()
        elif self.integrator.lower() == 'thug':
            self.verboseprint("Stochastic Kernel: THUG.")
            # Instantiate the class, doesn't matter which ξ0 or logpi we use.
            THUGSampler = TangentialHugSampler(self.manifold.sample(advanced=True), self.B*self.δ, self.B, self.N, 0.0, self.manifold.logprior, self.manifold.fullJacobian, method='linear', safe=True)
            self.MH_kernel = self.THUGSampler.mh_kernel
        else:
            raise ValueError("Unexpected value found for stochastic kernel.")

        # When ϵs are fixed and given, one must have a manual initialization.
        if (not self.adaptiveε) and self.initialization == 'prior':
            raise NotImplementedError("Currently, one can only use a manual initialization when using a fixed schedule of ϵs.")

        # Choose correct initialization proceedure (and sort out self.ϵs and self.ηs)
        if self.initialization == 'prior' and (self.adaptiveε):
            # Initializion from the prior can only be done when we adapt ϵ
            self.εs = []  # Will be filled with self.ϵmax calculated from prior samples
            self.log_ηs = []  # Will be filled with a log_ηϵmax
            self.initializer = lambda: self.manifold.sample_prior(self.N, seed=self.prior_seed)
        elif (self.initialization == 'manual'):
            if (self.ε0_manual is not None) and self.adaptiveε:
                # One must provide ε0_manual
                self.εs = [self.ε0_manual]
                self.log_ηs = [FilamentaryDistribution(self.manifold.generate_logηε, self.ε0_manual)]
                self.initializer = lambda: self.x0_manual
            elif (self.ε0_manual is None) and (not self.adaptiveε):
                # both z0 and ϵs are provided
                self.εs = self.εs_fixed
                self.log_ηs = [FilamentaryDistribution(self.manifold.generate_logηε, ε) for ε in self.εs]
                self.initializer = lambda: self.x0_manual
            else:
                raise ValueError("Invalid initialization specifications.")
        else:
            raise ValueError("Invalid initialization specifications.")

        # Choose correct distance function
        if self.univariate:
            self.compute_distances = self._compute_distances_univariate
        else:
            self.compute_distances = self._compute_distances_multivariate

        # Determine whether to switch RWM -> THUG or not
        if self.integrator.lower() == 'rwm_then_thug':
            self.switch = True
        else:
            self.switch = False

    def _compute_nth_tolerance(self, z):
        """If the εs schedule is fixed, this does nothing. However, if the schedule
        is adaptive (i.e. self.adaptiveε == True), then we compute it as a quantile
        of distances of the particles from the manifold."""
        if not self.adaptiveε:
            pass
        else:
            # compute distances
            distances = self.compute_distances(z) # SMC particles only include the position component
            # add distances to storage
            self.DISTANCES = vstack((self.DISTANCES, distances))
            # determine next ε as quantile of distances
            # do not use clip because otherwise we will never finish
            ε = min(self.εs[self.n-1], quantile(unique(distances), self.quantile_value))
            self.εs.append(ε)
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηε, ε))

    def _compute_weights(self, z):
        """Computes weights for SMC sampler using log-sum-exp trick."""
        logw = apply_along_axis(lambda z: self.log_ηs[self.n](z) - self.log_ηs[self.n-1](z), 1, z)
        W = exp(logw - logsumexp(logw))
        return W

    def _compute_distances_multivariate(self, x_particles):
        """Computes distances for a multivariate constraint function."""
        return norm(apply_along_axis(self.manifold.q, 1, x_particles), axis=1)

    def _compute_distances_univariate(self, x_particles):
        """Computes distances for a univariate constraint function."""
        return abs(apply_along_axis(self.manifold.q, 1, x_particles))

    def _update_δ(self):
        """If self.adaptiveδ is True, then we adapt based on proxy AP. Otherwise
        we keep it the same. Here there is no need to change M because we construct
        it at each iteration anyways."""
        if self.adaptiveδ:
            self.δ = clip(exp(log(self.δ) + 0.5*(self.APS[self.n] - self.pm_target)), self.δmin, self.δmax)
            self.δs.append(self.δ)
            self.verboseprint("\tStep-size adapted to: {:.16f}".format(self.δ))
        else:
            self.δs.append(self.δ)
            self.verboseprint("\tStep-size kept fixed at: {:.16f}".format(self.δ))

    def switch_kernel(self):
        """Switches from RWM to THUG kernels."""
        # the next 3 lines are taken verbatim from __init__ when integrator = 'THUG'
        # in the class initialization T, B, N, α, and logpi don't matter. Only thing that
        # matters is 'safe', 'fullJacobian', and 'method'.
        THUGSampler = TangentialHugSampler(self.manifold.sample(advanced=True), self.B*self.δ, self.B, self.N, 0.0, self.manifold.logprior, self.manifold.fullJacobian, method='linear', safe=True)
        self.MH_kernel = THUGSampler.mh_kernel
        # Store when the switch happend
        self.n_switch = self.n  # store when the switch happens
        self.switched = True
        self.verboseprint("\n")
        self.verboseprint("####################################")
        self.verboseprint("### SWITCHING TO THUG KERNEL ###")
        self.verboseprint("####################################")
        self.verboseprint("\n")

    def initialize_particles(self):
        """Initializes the particles and stores them in a separate variable, for checking purposes."""
        z0 = self.initializer()
        self.starting_particles = z0
        return z0

    def sample(self):
        """Samples using an SMC sampler.
        IMPORTANT: HERE THE PARTICLES CONSIST ONLY OF THE POSITIONS!!"""
        start_time = time()

        # INITIALIZE PARTICLES
        z = self.initialize_particles()               # (N, d)

        # STORAGE
        self.PARTICLES = z[None, ...]                 # (1, N, d)
        self.WEIGHTS   = full(self.N, 1 / self.N)     # (N, )
        self.ESS       = 1 / np.sum(self.WEIGHTS**2)  # (N, )
        self.DISTANCES = self.compute_distances(z)    # (N, )
        self.INDECES   = arange(self.N)               # (N, )
        self.APS       = [1.0]                        # Acceptance Probabilities
        self.δs = [self.δ]                            # Step sizes

        # If prior initialization, find εmax to start with
        if self.initialization == 'prior':
            self.εmax = np.max(self.DISTANCES)
            self.εs.append(self.εmax)
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηε, self.εmax))
            self.verboseprint("Setting initial epsilon to εmax = {:.16f}".format(self.εmax))

        self.n = 1
        try:
            while (self.n <= self.maxiter) and (abs(self.εs[self.n-1]) >= self.εmin) and (self.APS[self.n-1] >= self.min_pm):
                self.verboseprint("Iteration: ", self.n)

                # RESAMPLE PARTICLES
                if self.n == 1:
                    indeces = choice(a=arange(self.N), size=self.N, p=self.WEIGHTS)
                else:
                    indeces = choice(a=arange(self.N), size=self.N, p=self.WEIGHTS[self.n-1])
                self.INDECES = vstack((self.INDECES, indeces))
                z = self.PARTICLES[self.n-1][indeces, :]
                self.verboseprint("\tParticles resampled.")

                # SELECT TOLERANCE
                self._compute_nth_tolerance(z)
                self.verboseprint("\tEpsilon: {:.16f}".format(self.εs[self.n]))

                # COMPUTE WEIGHTS (resampling makes sure w = w_incremental)
                W = self._compute_weights(z)
                self.WEIGHTS = vstack((self.WEIGHTS, W))
                self.ESS     = vstack((self.ESS, 1 / np.sum(W**2)))
                self.verboseprint("\tWeights computed and normalised.")

                # MUTATION STEP (only propagate alive particles)
                M = lambda z: self.MH_kernel(z, self.B, self.δ, self.log_ηs[self.n])
                alive         = self.WEIGHTS[self.n] > 0.0
                alive_indeces = where(alive)[0]
                z_new         = deepcopy(z)
                for ix in alive_indeces:
                    z_new[ix] = M(z[ix])
                self.verboseprint("\tMutation step done.")

                # ESTIMATE ACCEPTANCE PROBABILITY
                ap_hat = 1 - (sum(np.all((z_new - z) == zeros(self.d), axis=1)) / self.N)
                z = z_new  # we called it z_new just to compute the AP
                self.APS.append(ap_hat)
                self.PARTICLES = vstack((self.PARTICLES, z[None, ...]))
                self.verboseprint("\tApprox AP: {:.16f}".format(ap_hat))

                # TUNE STEP SIZE
                self._update_δ()

                #### SWITCH KERNEL
                if self.integrator.lower() == 'rwm_then_thug':  # only happens when we allow switching
                    if not self.switched:                       # continue ahead only if we haven't already switched
                        if self.switch_strategy == 'εprop':
                            if self.n >= 2:
                                if ((self.εs[self.n-1] - self.εs[self.n]) / self.εs[self.n-1]) <= self.εprop_switch:
                                    self.switch_kernel()
                        elif self.switch_strategy == 'pm':
                            if self.APS[self.n] <= self.pm_switch:
                                self.switch_kernel()
                        else:
                            raise ValueError("Invalid switching strategy.")

                self.n += 1
            self.total_time = time() - start_time
        except (ValueError, KeyboardInterrupt) as e:
            print("Error was raised: ", e)
        return z


### Initialization Functions

# def init_RWMϵ0(MS):
#     """Initializes particles by using RWM to sample from ηϵ0. MS must be an instance
#     of the MarkovSnippets class."""
#     # Grab variables
#     x0       = MS.SETTINGS['ξ0']      # Initial point on Manifold
#     burnin   = MS.SETTINGS['burnin']
#     thinning = MS.SETTINGS['thinning']
#     N        = MS.SETTINGS['N']
#     δ0       = MS.SETTINGS['δ0']
#     ϵ0       = MS.ϵs[0]
#     logηϵ0   = MS.log_ηs[0]
#     d        = MS.d                    # Dimensionality of the x-component
#
#     # Sample (with burnin and thinning)
#     samples, acceptances = RWM(x0, s=δ0, N=(burnin + thinning*N), logpi=logηϵ0)
#     MS.verboseprint("Initializing particles from ηϵ0 with RWM. Acceptance: {:.2f}".format(np.mean(acceptances)*100))
#
#     # Thin samples to obtain initial particles
#     xparticles = samples[burnin:][::thinning]
#     vparticles = normal(loc=0.0, scale=1.0, size=(N, d))
#     z0 = hstack((xparticles, vparticles))
#     MS.starting_particles = z0
#     return z0

# def init_THUGϵ0(MS):
#     """Similar to init_RWMϵ0 but here we use THUG."""
#     # Grab variables
#     x0       = MS.SETTINGS['ξ0']      # Initial point on Manifold
#     burnin   = MS.SETTINGS['burnin']
#     thinning = MS.SETTINGS['thinning']
#     N        = MS.SETTINGS['N']
#     δ0       = MS.SETTINGS['δ0']
#     B        = MS.SETTINGS['B']
#     ϵ0       = MS.ϵs[0]
#     logηϵ0   = MS.log_ηs[0]
#     d        = MS.d                    # Dimensionality of the x-component
#
#     # Construct variables
#     q        = MVN(zeros(d), eye(d))
#
#     # Sample (with burnin and thinning)
#     samples, acceptances = HugTangentialMultivariate(
#         x0 = x0,
#         T  = B*δ0,
#         B  = B,
#         N  = (burnin+thinning*N),
#         α  = 0.0,
#         q  = q,
#         logpi = logηϵ0,
#         jac   = MS.manifold.fullJacobian,
#         method = 'linear')
#     MS.verboseprint("Initializing particles from ηϵ0 with THUG. Acceptance: {:.2f}".format(np.mean(acceptances)*100))
#
#     # Thin samples to obtain initial particles
#     xparticles = samples[burnin:][::thinning]
#     vparticles = normal(loc=0.0, scale=1.0, size=(N, d))
#     z0 = hstack((xparticles, vparticles))
#     MS.starting_particles = z0
#     return z0

# def init_on_manifold(MS):
#     """Samples points on manifold."""
#     # It would be too expensive to generate all particles,
#     # so instead we just generate some of them and then repeat them
#     n_initial_points = 200
#     initial_points = zeros((n_initial_points, MS.d))
#     MS.verboseprint("Initializing {} particles on manifold and repeating them.".format(n_initial_points))
#     for i in range(n_initial_points):
#         initial_points[i, :] = find_point_on_manifold(ystar=MS.manifold.ystar, ϵ=1e-8, max_iter=1000)
#     # Now repeat the array to achieve the number of particles
#     x0 = np.repeat(initial_points, repeats=((MS.N // n_initial_points) + 1), axis=0)[:MS.N, :]
#     # create tVelocities
#     v0 = np.random.rand(MS.N, MS.d)
#     z0 = np.hstack((x0, v0))
#     MS.starting_particles = z0
#     return z0


### Filamentary Distribution objects
class FilamentaryDistribution:

    def __init__(self, generate_logηϵ, ϵ):
        """Just a class object wrapping up a filamentary distribution."""
        self.ϵ = ϵ
        self.log_ηϵ = generate_logηϵ(ϵ)

    def __call__(self, ξ):
        """Simply computes the log-density of a filamentary distribution."""
        return self.log_ηϵ(ξ)

    def __repr__(self):
        """Pretty print"""
        return "Filamentary distribution with ϵ = {:.16f}".format(self.ϵ)
