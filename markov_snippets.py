"""
Various functions and classes for Markov Snippets.
"""
import numpy as np
from numpy import zeros, hstack, arange, repeat, log, concatenate, eye
from numpy import apply_along_axis, exp, unravel_index, dstack, vstack, ndarray
from numpy import quantile, unique
from numpy.linalg import solve, norm
from numpy.random import rand, choice, normal, randn
from scipy.stats import multivariate_normal as MVN
from time import time

from RWM import RWM
from tangential_hug_functions import HugTangentialMultivariate

### Markov Snippets classes

########### FIXED TOLERANCES, SINGLE INTEGRATOR ##################

class MSFixedTolerances:

    def __init__(self, SETTINGS):
        """Markov Snippets sampler that can choose between THUG or RWM integrators.
        Corresponds to Algorithm 1 in Christophe's notes. The sequence of target
        distributions is fixed here because we provide a fixed sequence of ϵs in
        the parameter SETTINGS.

        Tolerances
        ----------
        `SETTINGS['ϵs']` is of length
        `P+1`, meaning `ϵs = [ϵ0, ϵ1, ...., ϵP]` where ϵ0 is used to sample the
        initial N particles at the start.

        Initialization
        --------------
        We allow for multiple initialization procedures.
            1. init_RWMϵ0:  Starting from x0 ∈ Manifold, sample from ηϵ0 using RWM
                            with some thinning and some burn-in.
            2. init_THUGϵ0: Starting from x0 ∈ Manifold, sample from ηϵ0 using THUG
                            with some thinning and some burn-in.
            3. init_prior: Sample from the prior.

        Parameters
        ----------

        :param SETTINGS: Dictionary containing various variables necessary for
                         running the Markov Snippets algorithm.
        :type SETTINGS: dict
        """
        # Store variables
        self.N  = SETTINGS['N']       # Number of particles
        self.B  = SETTINGS['B']       # Number of integration steps
        self.δ  = SETTINGS['δ']       # Step-size for each integration step
        self.d  = SETTINGS['d']       # Dim of x-component of particle
        self.ϵs = SETTINGS['ϵs']      # Fixed schedule of tolerances
        self.manifold = SETTINGS['manifold']
        self.SETTINGS = SETTINGS
        self.thug     = SETTINGS['thug']
        self.verbose = SETTINGS['verbose']
        self.verboseprint = print if self.verbose else lambda *a, **k: None
        self.initialization = SETTINGS['initialization']

        # Check arguments
        assert isinstance(self.N,  int), "N must be an integer."
        assert isinstance(self.B, int), "B must be an integer."
        assert isinstance(self.δ, float), "δ must be a float."
        assert isinstance(self.d, int), "d must be an integer."
        assert isinstance(self.ϵs, list) or isinstance(self.ϵs, ndarray), "ϵs must be iterable."
        assert isinstance(self.thug, bool), "thug must be boolean variable."
        assert isinstance(self.initialization, str), "initialization must be a string."

        # Variables derived from the above
        self.P  = len(self.ϵs) - 1                                       # Number of target distributions
        self.log_ηs = [FilamentaryDistribution(self.manifold.generate_logηϵ, ϵ) for ϵ in self.ϵs] # List of filamentary distributions

        # Choose correct integrator based on user input
        if self.thug:
            self.verboseprint("Integrator: THUG.")
            self.ψ = generate_THUGIntegrator(self.B, self.δ, self.manifold.fullJacobian)
        else:
            self.verboseprint("Integrator: RWM.")
            self.ψ = generate_RWMIntegrator(self.B, self.δ)

        # Choose initialization procedure
        if self.initialization == 'init_RWMϵ0':
            self.initializer = init_RWMϵ0
        elif self.initialization == 'init_THUGϵ0':
            self.initializer = init_THUGϵ0
        elif self.initialization == 'init_prior':
            self.initializer = init_prior
        else:
            raise ValueError("Initializer must be one of three options.")

    def initialize_particles(self):
        """Initializes based on the user input. 3 options available, see docs."""
        z0 = self.initializer(self)
        return z0

    def sample(self):
        """Starts the Markov Snippets sampler."""
        starting_time = time()
        ## Storage
        #### Store z_n^{(i)}
        self.ZN  = zeros((self.P+1, self.N, 2*self.d))
        #### Store z_{n, k}^{(i)} so basically all the N(T+1) particles
        self.ZNK  = zeros((self.P, self.N*(self.B+1), 2*self.d))
        self.Wbar = zeros((self.P, self.N*(self.B+1)))
        self.ESS  = zeros((self.P))
        self.K_RESAMPLED = zeros((self.P, self.N))
        # Initialize particles
        z = self.initialize_particles()   # (N, 2d)
        self.ZN[0] = z
        # For each target distribution, run the following loop
        try:
            for n in range(1, self.P+1):
                self.verboseprint("Iteration: ", n, " Epsilon: {:.5f}".format(self.ϵs[n]))
                # Compute trajectories
                Z = apply_along_axis(self.ψ, 1, z)                      # (N, B+1, 2d)
                self.ZNK[n-1] = Z.reshape(self.N*(self.B+1), 2*self.d)  # (N(B+1), 2d)
                self.verboseprint("\tTrajectories constructed.")
                # Compute weights.
                #### Log-Denominator: shared for each point in the same trajectory
                log_μnm1_z  = apply_along_axis(self.log_ηs[n-1], 1, Z[:, 0, :self.d])        # (N, )
                log_μnm1_z  = repeat(log_μnm1_z, self.B+1, axis=0).reshape(self.N, self.B+1) # (N, B+1)
                #### Log-Numerator: different for each point on a trajectory.
                log_μn_ψk_z = apply_along_axis(self.log_ηs[n], 2, Z[:, :, :self.d])          # (N, B+1)
                #### Put weights together
                W = exp(log_μn_ψk_z - log_μnm1_z)                                            # (N, B+1)
                #### Normalize weights
                W = W / W.sum()
                self.verboseprint("\tWeights computed and normalized.")
                # store weights (remember these are \bar{w})
                self.Wbar[n-1] = W.flatten()
                # compute ESS
                self.ESS[n-1] = 1 / np.sum(W**2)
                # Resample down to N particles
                resampling_indeces = choice(a=arange(self.N*(self.B+1)), size=self.N, p=W.flatten())
                unravelled_indeces = unravel_index(resampling_indeces, (self.N, self.B+1))
                self.K_RESAMPLED[n-1] = unravelled_indeces[1]
                indeces = dstack(unravelled_indeces).squeeze()
                z = vstack([Z[tuple(ix)] for ix in indeces])     # (N, 2d)
                self.verboseprint("\tParticles Resampled.")

                # Rejuvenate velocities of N particles
                z[:, self.d:] = normal(loc=0.0, scale=1.0, size=(self.N, self.d))
                self.ZN[n] = z
                self.verboseprint("\tVelocities refreshed.")
            self.total_time = time() - starting_time
        except ValueError as e:
            print("ValueError was raised: ", e)
        return z

########## ADAPTIVE TOLERANCES, SINGLE INTEGRATOR #################

class MSAdaptiveTolerances:

    def __init__(self, SETTINGS):
        """Markov Snippets sampler that can choose between THUG or RWM integrators.
        Corresponds to Algorithm 1 in Christophe's notes. The sequence of target
        distributions is NOT fixed here. It is adaptively chosen based on the
        distribution of distances at the previous round, i.e. we choose a small
        quantile of all the distances.

        Tolerances
        ----------
        Adaptively chosen.

        Initialization
        --------------
        We allow for multiple initialization procedures.
            1. init_RWMϵ0:  Starting from x0 ∈ Manifold, sample from ηϵ0 using RWM
                            with some thinning and some burn-in.
            2. init_THUGϵ0: Starting from x0 ∈ Manifold, sample from ηϵ0 using THUG
                            with some thinning and some burn-in.
            3. init_prior: Sample from the prior.

        Parameters
        ----------

        :param SETTINGS: Dictionary containing various variables necessary for
                         running the Markov Snippets algorithm.
        :type SETTINGS: dict
        """
        # Store variables
        self.N  = SETTINGS['N']       # Number of particles
        self.B  = SETTINGS['B']       # Number of integration steps
        self.δ  = SETTINGS['δ']       # Step-size for each integration step
        self.d  = SETTINGS['d']       # Dim of x-component of particle
        self.manifold = SETTINGS['manifold']
        self.SETTINGS = SETTINGS
        self.thug     = SETTINGS['thug']
        self.verbose = SETTINGS['verbose']
        self.verboseprint = print if self.verbose else lambda *a, **k: None
        self.initialization = SETTINGS['initialization']
        self.ϵmin = SETTINGS['ϵmin']
        self.maxiter = SETTINGS['maxiter']
        self.quantile_value = SETTINGS['quantile_value']

        # Check arguments
        assert isinstance(self.N,  int), "N must be an integer."
        assert isinstance(self.B, int), "B must be an integer."
        assert isinstance(self.δ, float), "δ must be a float."
        assert isinstance(self.d, int), "d must be an integer."
        assert isinstance(self.thug, bool), "thug must be boolean variable."
        assert isinstance(self.initialization, str), "initialization must be a string."
        assert isinstance(self.ϵmin, float), "ϵmin must be a float."
        assert isinstance(self.maxiter, int), "maxiter must be an integer."
        assert isinstance(self.quantile_value, float), "quantile_value must be float."
        assert self.quantile_value >= 0 and self.quantile_value <= 1, "quantile value must be in [0, 1]."

        # Initialize the arrays storing ϵ and logηϵ as empty. If we initialize
        # from a small ϵ0 then we add it (and the corresponding logηϵ0) to the
        # list below. Otherwise, we consider it -np.inf and use the prior instead.
        self.ϵs     = []
        self.log_ηs = []

        # Choose correct integrator based on user input
        if self.thug:
            self.verboseprint("Integrator: THUG.")
            self.ψ = generate_THUGIntegrator(self.B, self.δ, self.manifold.fullJacobian)
        else:
            self.verboseprint("Integrator: RWM.")
            self.ψ = generate_RWMIntegrator(self.B, self.δ)

        # Choose initialization procedure
        if self.initialization == 'init_RWMϵ0':
            self.initializer = init_RWMϵ0
            self.ϵs.append(SETTINGS['ϵ0'])
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηϵ, self.ϵs[0]))
        elif self.initialization == 'init_THUGϵ0':
            self.initializer = init_THUGϵ0
            self.ϵs.append(SETTINGS['ϵ0'])
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηϵ, self.ϵs[0]))
        elif self.initialization == 'init_prior':
            self.initializer = init_prior
            self.ϵs.append(-np.inf)
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logprior, -np.inf))
        else:
            raise ValueError("Initializer must be one of three options.")

    def initialize_particles(self):
        """Initializes based on the user input. 3 options available, see docs."""
        z0 = self.initializer(self)
        return z0

    def sample(self):
        """Starts the Markov Snippets sampler."""
        starting_time = time()
        ## Storage
        #### Store z_n^{(i)}
        self.ZN  = zeros((1, self.N, 2*self.d))
        #### Store z_{n, k}^{(i)} so basically all the N(T+1) particles
        self.ZNK  = zeros((1, self.N*(self.B+1), 2*self.d))
        self.Wbar = zeros(self.N*(self.B+1))
        self.DISTANCES = zeros(self.N*(self.B+1))
        self.ESS  = [self.N]
        self.K_RESAMPLED = zeros(self.N)
        # Store proxy metrics for acceptance probabilities
        self.prop_moved = [] # Stores the proportion of particles with k >= 1
        # Initialize particles
        z = self.initialize_particles()   # (N, 2d)
        self.ZN[0] = z
        # Keep running until an error arises or we reach ϵ_min
        n = 1
        try:
            while n <= self.maxiter:
                self.verboseprint("Iteration: ", n)
                # Compute trajectories
                Z = apply_along_axis(self.ψ, 1, z)                                        # (N, B+1, 2d)
                self.ZNK = vstack((self.ZNK, Z.reshape(1, self.N*(self.B+1), 2*self.d)))  # (N(B+1), 2d)
                self.verboseprint("\tTrajectories constructed.")

                # Adaptively choose ϵ
                distances = norm(apply_along_axis(self.manifold.q, 1, self.ZNK[-1][:, :self.d]), axis=1)
                self.DISTANCES = vstack((self.DISTANCES, distances))
                ϵ = max(self.ϵmin, quantile(unique(distances), self.quantile_value))
                self.ϵs.append(ϵ)
                self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηϵ, ϵ))
                self.verboseprint("\tEpsilon: {:.6f}".format(ϵ))

                # Compute weights.
                #### Log-Denominator: shared for each point in the same trajectory
                log_μnm1_z  = apply_along_axis(self.log_ηs[-2], 1, Z[:, 0, :self.d])        # (N, )
                log_μnm1_z  = repeat(log_μnm1_z, self.B+1, axis=0).reshape(self.N, self.B+1) # (N, B+1)
                #### Log-Numerator: different for each point on a trajectory.
                log_μn_ψk_z = apply_along_axis(self.log_ηs[-1], 2, Z[:, :, :self.d])          # (N, B+1)
                #### Put weights together
                W = exp(log_μn_ψk_z - log_μnm1_z)                                            # (N, B+1)
                #### Normalize weights
                W = W / W.sum()
                self.verboseprint("\tWeights computed and normalized.")
                # store weights (remember these are \bar{w})
                self.Wbar = vstack((self.Wbar, W.flatten()))
                # compute ESS
                self.ESS.append(1 / np.sum(W**2))
                # Resample down to N particles
                resampling_indeces = choice(a=arange(self.N*(self.B+1)), size=self.N, p=W.flatten())
                unravelled_indeces = unravel_index(resampling_indeces, (self.N, self.B+1))
                self.K_RESAMPLED = vstack((self.K_RESAMPLED, unravelled_indeces[1]))
                indeces = dstack(unravelled_indeces).squeeze()
                z = vstack([Z[tuple(ix)] for ix in indeces])     # (N, 2d)
                self.verboseprint("\tParticles Resampled.")

                # Rejuvenate velocities of N particles
                z[:, self.d:] = normal(loc=0.0, scale=1.0, size=(self.N, self.d))
                self.ZN = vstack((self.ZN, z[None, ...]))
                self.verboseprint("\tVelocities refreshed.")

                # Compute proxy acceptance probabilities
                self.prop_moved.append(sum(self.K_RESAMPLED[-1] >= 1) / self.N)
                self.verboseprint("\tProp Moved: {:.3f}".format(self.prop_moved[-1]))

                n += 1
            self.total_time = time() - starting_time
        except ValueError as e:
            print("ValueError was raised: ", e)
        return z

######### ADAPTIVE TOLERANCES, SWITCH INTEGRATOR ################

class MSAdaptiveTolerancesSwitchIntegrator:

    def __init__(self, SETTINGS):
        """Markov Snippets sampler that starts with a RWM integrator and then switches
        to THUG once ϵ doesn't change more than a certain threshold. The sequence
        of target distributions is NOT fixed here. It is adaptively
        chosen based on the distribution of distances at the previous round, i.e.
        we choose a small quantile of all the distances.

        Tolerances
        ----------
        Adaptively chosen.

        Initialization
        --------------
        We allow for multiple initialization procedures.
            1. init_RWMϵ0:  Starting from x0 ∈ Manifold, sample from ηϵ0 using RWM
                            with some thinning and some burn-in.
            2. init_THUGϵ0: Starting from x0 ∈ Manifold, sample from ηϵ0 using THUG
                            with some thinning and some burn-in.
            3. init_prior: Sample from the prior.

        Parameters
        ----------

        :param SETTINGS: Dictionary containing various variables necessary for
                         running the Markov Snippets algorithm.
        :type SETTINGS: dict
        """
        # Store variables
        self.N  = SETTINGS['N']       # Number of particles
        self.B  = SETTINGS['B']       # Number of integration steps
        self.δ  = SETTINGS['δ']       # Step-size for each integration step
        self.d  = SETTINGS['d']       # Dim of x-component of particle
        self.manifold = SETTINGS['manifold']
        self.SETTINGS = SETTINGS
        self.verbose = SETTINGS['verbose']
        self.verboseprint = print if self.verbose else lambda *a, **k: None
        self.initialization = SETTINGS['initialization']
        self.ϵmin = SETTINGS['ϵmin']
        self.maxiter = SETTINGS['maxiter']
        self.quantile_value = SETTINGS['quantile_value']
        self.switch_strategy = SETTINGS['switch_strategy']
        self.ϵprop_switch = SETTINGS['ϵprop_switch'] # once below this, switch to THUG
        self.pmoved_switch = SETTINGS['pmoved_switch']

        # Check arguments
        assert isinstance(self.N,  int), "N must be an integer."
        assert isinstance(self.B, int), "B must be an integer."
        assert isinstance(self.δ, float), "δ must be a float."
        assert isinstance(self.d, int), "d must be an integer."
        assert isinstance(self.initialization, str), "initialization must be a string."
        assert isinstance(self.ϵmin, float), "ϵmin must be a float."
        assert isinstance(self.maxiter, int), "maxiter must be an integer."
        assert isinstance(self.quantile_value, float), "quantile_value must be float."
        assert self.quantile_value >= 0 and self.quantile_value <= 1, "quantile value must be in [0, 1]."
        assert isinstance(self.ϵprop_switch, float), "ap_switch must be float."
        assert self.ϵprop_switch >=0 and self.ϵprop_switch <= 1, "ap_switch must be in [0, 1]."
        assert isinstance(self.pmoved_switch, float), "pmoved_switch must be float."
        assert self.pmoved_switch >=0 and self.pmoved_switch <= 1, "pmoved_switch must be in [0, 1]."
        assert self.switch_strategy in ['ϵprop', 'ap'], "switch strategy must be either `ϵprop` or `ap`."

        # Initialize the arrays storing ϵ and logηϵ as empty. If we initialize
        # from a small ϵ0 then we add it (and the corresponding logηϵ0) to the
        # list below. Otherwise, we consider it -np.inf and use the prior instead.
        self.ϵs     = []
        self.log_ηs = []
        self.switched = False

        # Start with RWM integrator, then switch to THUG
        self.verboseprint("Integrator: RWM. Switching strategy: ", self.switch_strategy)
        self.ψ      = generate_RWMIntegrator(self.B, self.δ)
        self.ψ_thug = generate_THUGIntegrator(self.B, self.δ, self.manifold.fullJacobian)

        # Choose initialization procedure
        if self.initialization == 'init_RWMϵ0':
            self.initializer = init_RWMϵ0
            self.ϵs.append(SETTINGS['ϵ0'])
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηϵ, self.ϵs[0]))
        elif self.initialization == 'init_THUGϵ0':
            self.initializer = init_THUGϵ0
            self.ϵs.append(SETTINGS['ϵ0'])
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηϵ, self.ϵs[0]))
        elif self.initialization == 'init_prior':
            self.initializer = init_prior
            self.ϵs.append(-np.inf)
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logprior, -np.inf))
        else:
            raise ValueError("Initializer must be one of three options.")

    def initialize_particles(self):
        """Initializes based on the user input. 3 options available, see docs."""
        z0 = self.initializer(self)
        return z0

    def switch_integrator(self, n):
        """Switches from RWM to THUG."""
        self.ψ = self.ψ_thug
        self.n_switch = n  # store when the switch happens
        self.switched = True
        self.verboseprint("####################################")
        self.verboseprint("### SWITCHING TO THUG INTEGRATOR ###")
        self.verboseprint("####################################")

    def sample(self):
        """Starts the Markov Snippets sampler."""
        starting_time = time()
        ## Storage
        #### Store z_n^{(i)}
        self.ZN  = zeros((1, self.N, 2*self.d))
        #### Store z_{n, k}^{(i)} so basically all the N(T+1) particles
        self.ZNK  = zeros((1, self.N*(self.B+1), 2*self.d))
        self.Wbar = zeros(self.N*(self.B+1))
        self.DISTANCES = zeros(self.N*(self.B+1))
        self.ESS  = [self.N]
        self.K_RESAMPLED = zeros(self.N)
        # Store proxy metrics for acceptance probabilities
        self.prop_moved = [] # Stores the proportion of particles with k >= 1
        # Initialize particles
        z = self.initialize_particles()   # (N, 2d)
        self.ZN[0] = z
        # Keep running until an error arises or we reach ϵ_min
        n = 1
        try:
            while n <= self.maxiter:
                self.verboseprint("Iteration: ", n)
                # Compute trajectories
                Z = apply_along_axis(self.ψ, 1, z)                                        # (N, B+1, 2d)
                self.ZNK = vstack((self.ZNK, Z.reshape(1, self.N*(self.B+1), 2*self.d)))  # (N(B+1), 2d)
                self.verboseprint("\tTrajectories constructed.")

                # Adaptively choose ϵ
                distances = norm(apply_along_axis(self.manifold.q, 1, self.ZNK[-1][:, :self.d]), axis=1)
                self.DISTANCES = vstack((self.DISTANCES, distances))
                ϵ = max(self.ϵmin, quantile(unique(distances), self.quantile_value))
                self.ϵs.append(ϵ)
                self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηϵ, ϵ))
                self.verboseprint("\tEpsilon: {:.6f}".format(ϵ))

                # Compute weights.
                #### Log-Denominator: shared for each point in the same trajectory
                log_μnm1_z  = apply_along_axis(self.log_ηs[-2], 1, Z[:, 0, :self.d])        # (N, )
                log_μnm1_z  = repeat(log_μnm1_z, self.B+1, axis=0).reshape(self.N, self.B+1) # (N, B+1)
                #### Log-Numerator: different for each point on a trajectory.
                log_μn_ψk_z = apply_along_axis(self.log_ηs[-1], 2, Z[:, :, :self.d])          # (N, B+1)
                #### Put weights together
                W = exp(log_μn_ψk_z - log_μnm1_z)                                            # (N, B+1)
                #### Normalize weights
                W = W / W.sum()
                self.verboseprint("\tWeights computed and normalized.")
                # store weights (remember these are \bar{w})
                self.Wbar = vstack((self.Wbar, W.flatten()))
                # compute ESS
                self.ESS.append(1 / np.sum(W**2))
                # Resample down to N particles
                resampling_indeces = choice(a=arange(self.N*(self.B+1)), size=self.N, p=W.flatten())
                unravelled_indeces = unravel_index(resampling_indeces, (self.N, self.B+1))
                self.K_RESAMPLED = vstack((self.K_RESAMPLED, unravelled_indeces[1]))
                indeces = dstack(unravelled_indeces).squeeze()
                z = vstack([Z[tuple(ix)] for ix in indeces])     # (N, 2d)
                self.verboseprint("\tParticles Resampled.")

                # Rejuvenate velocities of N particles
                z[:, self.d:] = normal(loc=0.0, scale=1.0, size=(self.N, self.d))
                self.ZN = vstack((self.ZN, z[None, ...]))
                self.verboseprint("\tVelocities refreshed.")

                # Compute proxy acceptance probabilities
                self.prop_moved.append(sum(self.K_RESAMPLED[-1] >= 1) / self.N)
                self.verboseprint("\tProp Moved: {:.3f}".format(self.prop_moved[-1]))

                # Check if it's time to switch to THUG integrator
                # we need to check n >= 2 because if we initialize from the prior
                # then -np.inf at the denominator would blow everything up
                if not self.switched:
                    if self.switch_strategy == 'ϵprop':
                        if (n >= 2):
                            if ((self.ϵs[-2] - self.ϵs[-1]) / self.ϵs[-2]) <= self.ϵprop_switch:
                                self.switch_integrator(n)
                    else:
                        if self.prop_moved[-1] <= self.pmoved_switch:
                            self.switch_integrator(n)

                n += 1
            self.total_time = time() - starting_time
        except ValueError as e:
            print("ValueError was raised: ", e)
        return z

######### ADAPTIVE TOLERANCES, SINGLE INTEGRATOR, ADAPTIVE STEP SIZE ##########

class MSAdaptiveTolerancesAdaptiveδ:

    def __init__(self, SETTINGS):
        """Markov Snippets sampler that can choose between THUG or RWM integrators.
        The sequence of target distributions is NOT fixed here. It is adaptively
        chosen based on the  distribution of distances at the previous round, i.e.
        we choose a small quantile of all the distances.
        The step size is chosen adaptively like in a standard SMC sampler (a la
        Chang) using the proxy for the acceptance probability.

        Tolerances
        ----------
        Adaptively chosen.

        Initialization
        --------------
        We allow for multiple initialization procedures.
            1. init_RWMϵ0:  Starting from x0 ∈ Manifold, sample from ηϵ0 using RWM
                            with some thinning and some burn-in.
            2. init_THUGϵ0: Starting from x0 ∈ Manifold, sample from ηϵ0 using THUG
                            with some thinning and some burn-in.
            3. init_prior: Sample from the prior.

        Parameters
        ----------

        :param SETTINGS: Dictionary containing various variables necessary for
                         running the Markov Snippets algorithm.
        :type SETTINGS: dict
        """
        # Store variables
        self.N  = SETTINGS['N']       # Number of particles
        self.B  = SETTINGS['B']       # Number of integration steps
        self.δ  = SETTINGS['δ']       # Step-size for each integration step
        self.d  = SETTINGS['d']       # Dim of x-component of particle
        self.manifold = SETTINGS['manifold']
        self.SETTINGS = SETTINGS
        self.thug     = SETTINGS['thug']
        self.verbose = SETTINGS['verbose']
        self.verboseprint = print if self.verbose else lambda *a, **k: None
        self.initialization = SETTINGS['initialization']
        self.ϵmin = SETTINGS['ϵmin']
        self.maxiter = SETTINGS['maxiter']
        self.quantile_value = SETTINGS['quantile_value']
        self.ap_target = SETTINGS['ap_target']  # target acceptance probability used to adapt δ
        self.δmin = SETTINGS['δmin']
        self.δmax = SETTINGS['δmax'] # both used for adaptation

        # Check arguments
        assert isinstance(self.N,  int), "N must be an integer."
        assert isinstance(self.B, int), "B must be an integer."
        assert isinstance(self.δ, float), "δ must be a float."
        assert isinstance(self.d, int), "d must be an integer."
        assert isinstance(self.thug, bool), "thug must be boolean variable."
        assert isinstance(self.initialization, str), "initialization must be a string."
        assert isinstance(self.ϵmin, float), "ϵmin must be a float."
        assert isinstance(self.maxiter, int), "maxiter must be an integer."
        assert isinstance(self.quantile_value, float), "quantile_value must be float."
        assert self.quantile_value >= 0 and self.quantile_value <= 1, "quantile value must be in [0, 1]."
        assert isinstance(self.ap_target, float), "ap_target must be float."
        assert self.ap_target >= 0 and self.ap_target <= 1, "ap_target must be in [0, 1]."
        assert isinstance(self.δmin, float), "δmin must be float."
        assert isinstance(self.δmax, float), "δmax, must be float."
        assert (0 <= self.δmin) and (self.δmin <= self.δmax), "step sizes must be positive and ordered."



        # Initialize the arrays storing ϵ and logηϵ as empty. If we initialize
        # from a small ϵ0 then we add it (and the corresponding logηϵ0) to the
        # list below. Otherwise, we consider it -np.inf and use the prior instead.
        self.ϵs     = []
        self.log_ηs = []
        self.δs     = SETTINGS['δ']

        # Choose correct integrator based on user input
        if self.thug:
            self.verboseprint("Integrator: THUG.")
            self.ψ = generate_THUGIntegrator(self.B, self.δ, self.manifold.fullJacobian)
            self.ψ_generator = lambda B, δ: generate_THUGIntegrator(B, δ, self.manifold.fullJacobian)
        else:
            self.verboseprint("Integrator: RWM.")
            self.ψ = generate_RWMIntegrator(self.B, self.δ)
            self.ψ_generator = generate_RWMIntegrator

        # Choose initialization procedure
        if self.initialization == 'init_RWMϵ0':
            self.initializer = init_RWMϵ0
            self.ϵs.append(SETTINGS['ϵ0'])
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηϵ, self.ϵs[0]))
        elif self.initialization == 'init_THUGϵ0':
            self.initializer = init_THUGϵ0
            self.ϵs.append(SETTINGS['ϵ0'])
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηϵ, self.ϵs[0]))
        elif self.initialization == 'init_prior':
            self.initializer = init_prior
            self.ϵs.append(-np.inf)
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logprior, -np.inf))
        else:
            raise ValueError("Initializer must be one of three options.")

    def initialize_particles(self):
        """Initializes based on the user input. 3 options available, see docs."""
        z0 = self.initializer(self)
        return z0

    def sample(self):
        """Starts the Markov Snippets sampler."""
        starting_time = time()
        ## Storage
        #### Store z_n^{(i)}
        self.ZN  = zeros((1, self.N, 2*self.d))
        #### Store z_{n, k}^{(i)} so basically all the N(T+1) particles
        self.ZNK  = zeros((1, self.N*(self.B+1), 2*self.d))
        self.Wbar = zeros(self.N*(self.B+1))
        self.DISTANCES = zeros(self.N*(self.B+1))
        self.ESS  = [self.N]
        self.K_RESAMPLED = zeros(self.N)
        # Store proxy metrics for acceptance probabilities
        self.prop_moved = [] # Stores the proportion of particles with k >= 1
        # Initialize particles
        z = self.initialize_particles()   # (N, 2d)
        self.ZN[0] = z
        # Keep running until an error arises or we reach ϵ_min
        n = 1
        try:
            while n <= self.maxiter:
                self.verboseprint("Iteration: ", n)
                # Compute trajectories
                Z = apply_along_axis(self.ψ, 1, z)                                        # (N, B+1, 2d)
                self.ZNK = vstack((self.ZNK, Z.reshape(1, self.N*(self.B+1), 2*self.d)))  # (N(B+1), 2d)
                self.verboseprint("\tTrajectories constructed.")

                # Adaptively choose ϵ
                distances = norm(apply_along_axis(self.manifold.q, 1, self.ZNK[-1][:, :self.d]), axis=1)
                self.DISTANCES = vstack((self.DISTANCES, distances))
                ϵ = max(self.ϵmin, quantile(unique(distances), self.quantile_value))
                self.ϵs.append(ϵ)
                self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηϵ, ϵ))
                self.verboseprint("\tEpsilon: {:.6f}".format(ϵ))

                # Compute weights.
                #### Log-Denominator: shared for each point in the same trajectory
                log_μnm1_z  = apply_along_axis(self.log_ηs[-2], 1, Z[:, 0, :self.d])        # (N, )
                log_μnm1_z  = repeat(log_μnm1_z, self.B+1, axis=0).reshape(self.N, self.B+1) # (N, B+1)
                #### Log-Numerator: different for each point on a trajectory.
                log_μn_ψk_z = apply_along_axis(self.log_ηs[-1], 2, Z[:, :, :self.d])          # (N, B+1)
                #### Put weights together
                W = exp(log_μn_ψk_z - log_μnm1_z)                                            # (N, B+1)
                #### Normalize weights
                W = W / W.sum()
                self.verboseprint("\tWeights computed and normalized.")
                # store weights (remember these are \bar{w})
                self.Wbar = vstack((self.Wbar, W.flatten()))
                # compute ESS
                self.ESS.append(1 / np.sum(W**2))
                # Resample down to N particles
                resampling_indeces = choice(a=arange(self.N*(self.B+1)), size=self.N, p=W.flatten())
                unravelled_indeces = unravel_index(resampling_indeces, (self.N, self.B+1))
                self.K_RESAMPLED = vstack((self.K_RESAMPLED, unravelled_indeces[1]))
                indeces = dstack(unravelled_indeces).squeeze()
                z = vstack([Z[tuple(ix)] for ix in indeces])     # (N, 2d)
                self.verboseprint("\tParticles Resampled.")

                # Rejuvenate velocities of N particles
                z[:, self.d:] = normal(loc=0.0, scale=1.0, size=(self.N, self.d))
                self.ZN = vstack((self.ZN, z[None, ...]))
                self.verboseprint("\tVelocities refreshed.")

                # Compute proxy acceptance probabilities
                self.prop_moved.append(sum(self.K_RESAMPLED[-1] >= 1) / self.N)
                self.verboseprint("\tProp Moved: {:.3f}".format(self.prop_moved[-1]))

                # Adapt δ based on proxy acceptance probability
                self.δ = clip(exp(log(self.δ) + 0.5*(self.prop_moved - self.ap_target)), self.δmin, self.δmax)
                self.δs.append(self.δ)
                self.ψ = self.ψ_generator(self.B, self.δ)
                self.verboseprint("\tStep-size adapted to: {:.8f}".format(self.δ))

                n += 1
            self.total_time = time() - starting_time
        except ValueError as e:
            print("ValueError was raised: ", e)
        return z








### Integrator functions

def THUGIntegrator(z0, B, δ, jacobian_function):
    """Tangential Hug integrator. Given z0 = [x0, v0] as a starting point,
    it constructs a deterministic trajectory with B bounces of step-size δ.
    It outputs a trajectory, i.e. an array of size (B+1, 2d) where d is the
    dimensionality of the x-coordinate (and v-coordinate) and where the ith
    row, `trajectory[i, :]` corresponds to `[x_i, v_i]`. Uses a linear
    projection function."""
    # Set up
    linear_project = lambda v, J: J.T.dot(solve(J.dot(J.T), J.dot(v)))
    trajectory = zeros((B + 1, len(z0)))
    x0, v0 = z0[:len(z0)//2], z0[len(z0)//2:]
    x, v = x0, v0
    trajectory[0, :] = z0
    # Integrate
    for b in range(B):
        x = x + δ*v/2
        v = v - 2*linear_project(v, jacobian_function(x))
        x = x + δ*v/2
        trajectory[b+1, :] = hstack((x, v))
    return trajectory

def generate_THUGIntegrator(B, δ, jacobian_function):
    """Returns a THUG integrator for a given B and δ."""
    # integrator = lambda z: THUGIntegrator(z, B, δ, jacobian_function)
    class THUGIntegratorClass:
        def __init__(self, B, δ, jacobian_function):
            self.B   = B
            self.δ   = δ
            self.jac = jacobian_function

        def __repr__(self):
            return "THUG Integrator with B = {} and δ = {:.6f}".format(self.B, self.δ)

        def __call__(self, z):
            integrator = lambda z: THUGIntegrator(z, self.B, self.δ, self.jac)
            return integrator(z)
    return THUGIntegratorClass(B, δ, jacobian_function)

def RWMIntegrator(z0, B, δ):
    """Random Walk integrator. Given z0 = [x0, v0] it constructs a RW trajectory
    with B steps of step-size δ. This is a deterministic trajectory, and since
    RWM does not use gradients, this corresponds to B+1 points in a straight line
    starting from x0 and in the direction of v0."""
    trajectory = zeros((B+1, len(z0)))
    x0, v0 = z0[:len(z0)//2], z0[len(z0)//2:]
    bs  = arange(B+1).reshape(-1, 1) # 0, 1, ..., B
    xbs = x0 + δ*bs*v0     # move them by b*δ
    vbs = repeat(v0.reshape(1, -1), repeats=B+1, axis=0)
    zbs = hstack((xbs, vbs))
    return zbs

def generate_RWMIntegrator(B, δ):
    """Generates the integrator above."""
    # integrator = lambda z: RWMIntegrator(z, B, δ)
    class RWMIntegratorClass:
        def __init__(self, B, δ):
            self.B   = B
            self.δ   = δ

        def __repr__(self):
            return "RWM Integrator with B = {} and δ = {:.6f}".format(self.B, self.δ)

        def __call__(self, z):
            integrator = lambda z: RWMIntegrator(z, self.B, self.δ)
            return integrator(z)
    return RWMIntegratorClass(B, δ)

def THUG_MH(z0, B, δ, logpi, jacobian_function):
    """Similar to THUGIntegrator but this uses a Metropolis-Hastings step at each
    step, meaning that it is not deterministic. Given z0 = [x0, v0] it constructs
    the trajectory and then either accepts the final point zB or accepts the
    initial point z0.
    Notice this returns a single z = (x, v), not the whole trajectory."""
    x0, v0 = z0[:len(z0)//2], z0[len(z0)//2:]
    x, v = x0, v0
    logu = log(rand())
    for _ in range(B):
        x = x + δ*v/2
        v = v - 2*linear_project(v, jacobian_function(x))
        x = x + δ*v/2
    if logu <= logpi(x) - logpi(x0):
        return concatenate((x, v))    # accept new point
    else:
        return z0                     # accept old point

def RWM_MH(z0, B, δ, logpi):
    """Similar to RWMIntegrator but this uses a MH step. Again, this returns
    either the starting point z0 or the final point zB, not the entire
    trajectory. Importantly, the whole trajectory can be computed at once since
    it is linear, so instead of performing B linear trajectories of step δ, we
    simply perform a single step of size Bδ. """
    x0, v0 = z0[:len(z0)//2], z0[len(z0)//2:]
    logu = log(rand())
    x_new = x0 + B*δ*v0
    if logu <= logpi(x_new) - logpi(x0):
        return concatenate((x_new, v0))  # accept new position, old velocity
    else:
        return z0                        # accept old position and old velocity


### Initialization Functions

def init_RWMϵ0(MS):
    """Initializes particles by using RWM to sample from ηϵ0. MS must be an instance
    of the MarkovSnippets class."""
    # Grab variables
    x0       = MS.SETTINGS['ξ0']      # Initial point on Manifold
    burnin   = MS.SETTINGS['burnin']
    thinning = MS.SETTINGS['thinning']
    N        = MS.SETTINGS['N']
    δ0       = MS.SETTINGS['δ0']
    ϵ0       = MS.ϵs[0]
    logηϵ0   = MS.log_ηs[0]
    d        = MS.d                    # Dimensionality of the x-component

    # Sample (with burnin and thinning)
    samples, acceptances = RWM(x0, s=δ0, N=(burnin + thinning*N), logpi=logηϵ0)
    MS.verboseprint("Initializing particles from ηϵ0 with RWM. Acceptance: {:.2f}".format(np.mean(acceptances)*100))

    # Thin samples to obtain initial particles
    xparticles = samples[burnin:][::thinning]
    vparticles = normal(loc=0.0, scale=1.0, size=(N, d))
    z0 = hstack((xparticles, vparticles))
    MS.starting_particles = z0
    return z0

def init_THUGϵ0(MS):
    """Similar to init_RWMϵ0 but here we use THUG."""
    # Grab variables
    x0       = MS.SETTINGS['ξ0']      # Initial point on Manifold
    burnin   = MS.SETTINGS['burnin']
    thinning = MS.SETTINGS['thinning']
    N        = MS.SETTINGS['N']
    δ0       = MS.SETTINGS['δ0']
    B        = MS.SETTINGS['B']
    ϵ0       = MS.ϵs[0]
    logηϵ0   = MS.log_ηs[0]
    d        = MS.d                    # Dimensionality of the x-component

    # Construct variables
    q        = MVN(zeros(d), eye(d))

    # Sample (with burnin and thinning)
    samples, acceptances = HugTangentialMultivariate(
        x0 = x0,
        T  = B*δ0,
        B  = B,
        N  = (burnin+thinning*N),
        α  = 0.0,
        q  = q,
        logpi = logηϵ0,
        jac   = MS.manifold.fullJacobian,
        method = 'linear')
    MS.verboseprint("Initializing particles from ηϵ0 with THUG. Acceptance: {:.2f}".format(np.mean(acceptances)*100))

    # Thin samples to obtain initial particles
    xparticles = samples[burnin:][::thinning]
    vparticles = normal(loc=0.0, scale=1.0, size=(N, d))
    z0 = hstack((xparticles, vparticles))
    MS.starting_particles = z0
    return z0

def init_prior(MS):
    """Samples particles from the prior using RWM."""
    # Notice that the prior for the G and K problem is simply N(0, I)
    # and so is the distribution of the velocities, so this should do the job
    z0 = randn(MS.N, 2*MS.d)
    MS.verboseprint("Initializing particles from prior.")
    MS.starting_particles = z0
    return z0


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
        return "Filamentary distribution with ϵ = {:.8f}".format(self.ϵ)
