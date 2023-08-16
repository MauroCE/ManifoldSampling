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
import multiprocessing #from pathos.multiprocessing import ProcessingPool as Pool #from multiprocessing import Pool
from itertools import product

from RWM import RWM, generate_RWMIntegrator
# from tangential_hug_functions import HugTangentialMultivariate
# from tangential_hug_functions import HugTangential
from Manifolds.Manifold import Manifold
from tangential_hug_functions import TangentialHugSampler
from Manifolds.GKManifoldNew import find_point_on_manifold


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
        self.δ    = SETTINGS['δ']          # Step-size for each integration step, can be either float or 2 steps sizes if RWM and THUG are both used
        self.d    = SETTINGS['d']          # Dim of x-component of particle
        self.δmin = SETTINGS['δmin']       # δ ≧ δmin when adaptive
        self.δmax = SETTINGS['δmax']       # δ ≦ δmax when step size is adaptive
        self.εmin = SETTINGS['εmin']       # ε ≧ εmin when tolerance is adaptive
        self.Bmin = SETTINGS['Bmin']       # minimum number of steps, used when adaptiveB=True
        self.Bmax = SETTINGS['Bmax']       # maximum number of steps, used when adaptiveB=False
        self.min_pm = SETTINGS['min_pm'] # if fewer resampled particles have k ≧ 1 than self.min_pm we stop the algorithm (pm stands for prop-moved)
        self.maxiter = SETTINGS['maxiter'] # MS stopped when n ≧ maxiter
        self.verbose = SETTINGS['verbose'] # Whether to print MS progress
        self.NBbudget = SETTINGS['NBbudget'] # budget for NB adaptation
        self.prop_hug = SETTINGS['prop_hug'] # when using hug_and_nhug, proportion of trajectory generated with hug
        self.εs_fixed = SETTINGS['εs_fixed']   # Sequence of tolerances, this is only used if adaptiveϵ is False
        self.manifold = SETTINGS['manifold'] # Manifold around which we sample
        self.prop_esjd = SETTINGS['prop_esjd'] # proportion used when adapting δ based on ESJD
        self.ε0_manual = SETTINGS['ε0_manual'] # When initialization is 'manual' then this is the ϵ0 that we assume it has been sampled from
        self.adaptiveε = SETTINGS['adaptiveε'] # If true, we adapt ε based on the distances, otherwise we expect a fixed sequence
        self.adaptiveδ = SETTINGS['adaptiveδ'] # If true, we adapt δ based on the proxy acceptance probability.
        self.adaptiveB = SETTINGS['adaptiveB'] # if true, we adapt B. for the moment we only allow one type of adaptation: ESJD+PM
        self.adaptiveN = SETTINGS['adaptiveN'] # used exclusively when we only change B, for now
        self.z0_manual = SETTINGS['z0_manual'] # Starting particles, used only if initialization is manual
        self.pm_target = SETTINGS['pm_target'] # Target 'proportion of moved' particles the we aim to while adapting δ
        self.pd_target = SETTINGS['pd_target'] # Target particle diversity
        self.dm_target = SETTINGS['dm_target'] # Target diversity moved
        self.pm_switch = SETTINGS['pm_switch'] # when the proportion of particles moved is less than this, we switch
        self.prior_seed = SETTINGS['prior_seed'] # Seed to used when initializing the particles from the prior, to allow reproducibility
        self.low_memory = SETTINGS['low_memory'] # whether to use a low-memory version or not. Low memory does not store ZNK
        self.integrator = SETTINGS['integrator'] # Determines if we use RWM, THUG or start with RWM and switch to THUG
        self.εprop_switch = SETTINGS['εprop_switch'] # When (ε_n - ε_{n-1}) / ε_n is less than self.εprop_switch, then we switch
        self.metropolised = SETTINGS['metropolised'] # whether to keep whole trajectory (False) or only start and end (True)
        self.quantile_value = SETTINGS['quantile_value'] # Used to determine the next ϵ
        self.initialization = SETTINGS['initialization'] # type of initialization to use
        self.proxy_ap_metric = SETTINGS['proxy_ap_metric'] # which metric to use to adaptδ (pm, pd, or md)
        self.switch_strategy = SETTINGS['switch_strategy'] # strategy used to determine when to switch RWM->THUG.
        self.resampling_seed = SETTINGS['resampling_seed'] # seed for resampling
        self.resampling_scheme = SETTINGS['resampling_scheme'] # resampling scheme to use
        self.projection_method = SETTINGS['projection_method'] # method used in THUG to project ('linear' or 'qr')
        self.stopping_criterion = SETTINGS['stopping_criterion'] # determines strategy used to terminate the algorithm
        self.δadaptation_method = SETTINGS['δadaptation_method'] # determines adaptation method of δ

        # Check arguments types
        assert isinstance(self.N,  int), "N must be an integer."
        assert isinstance(self.B, int), "B must be an integer."
        assert isinstance(self.δ, float) or isinstance(self.δ, list), "δ must be a float or a list."
        assert isinstance(self.d, int), "d must be an integer."
        assert isinstance(self.δmin, float), "δmin must be float."
        assert isinstance(self.δmax, float), "δmax must be float."
        assert isinstance(self.εmin, float), "εmin must be float."
        assert isinstance(self.Bmin, int), "Bmin must be float."
        assert isinstance(self.Bmax, int), "Bmax must be float."
        assert isinstance(self.min_pm, float), "min_pm must be float."
        assert isinstance(self.maxiter, int), "maxiter must be integer."
        assert isinstance(self.verbose, bool), "verbose must be boolean."
        assert (self.NBbudget is None) or isinstance(self.NBbudget, int), "NBbudget must be None or int."
        assert isinstance(self.prop_hug, float), "prop_hug must be float."
        assert isinstance(self.εs_fixed, np.ndarray) or (self.εs_fixed is None), "εs must be a numpy array or must be None."
        assert isinstance(self.manifold, Manifold), "manifold must be an instance of class Manifold."
        assert isinstance(self.prop_esjd, float), "prop_esjd must be a float."
        assert isinstance(self.adaptiveε, bool), "adaptiveϵ must be bool."
        assert isinstance(self.adaptiveδ, bool), "adaptiveδ must be bool."
        assert isinstance(self.adaptiveB, bool), "adaptiveB must be bool."
        assert isinstance(self.adaptiveN, bool), "adaptiveN must be bool."
        assert isinstance(self.z0_manual, np.ndarray) or (self.z0_manual is None), "z0_manual must be a numpy array or None."
        assert isinstance(self.pm_target, float) or isinstance(self.pm_target, list), "pm_target must be float or list."
        assert isinstance(self.pd_target, float) or isinstance(self.pd_target, list), "pd_target must be float or list."
        assert isinstance(self.dm_target, float) or isinstance(self.dm_target, list), "dm_target must be float or list."
        assert isinstance(self.pm_switch, float), "pm_switch must be float."
        assert isinstance(self.prior_seed, int), "prior_seed must be integer."
        assert isinstance(self.low_memory, bool), "low_memory must be bool."
        assert isinstance(self.integrator, str), "integrator must be a string."
        assert isinstance(self.εprop_switch, float), "εprop_switch must be a float."
        assert isinstance(self.metropolised, bool), "metropolised must be bool."
        assert (self.ε0_manual is None) or isinstance(self.ε0_manual, float), "ε0_manual must be float or None."
        assert isinstance(self.quantile_value, float) or isinstance(self.quantile_value, list), "quantile_value must be a float or list."
        assert isinstance(self.proxy_ap_metric, str), "proxy_ap_metric must be string."
        assert isinstance(self.initialization, str), "initialization must be a string."
        assert isinstance(self.switch_strategy, str), "switch_strategy must be a string."
        assert isinstance(self.resampling_seed, int), "resampling seed must be integer."
        assert isinstance(self.resampling_scheme, str), "resampling_scheme must be a string."
        assert isinstance(self.projection_method, str), "projection_method must be a string."
        assert isinstance(self.stopping_criterion, set), "stopping criterion must be a set."
        assert (self.δadaptation_method is None) or isinstance(self.δadaptation_method, str), "δadaptation_method must be a string or None."


        # Check argument values
        if isinstance(self.δ, float):
            assert self.δ > 0.0, "δ must be larger than 0."
        elif isinstance(self.δ, list) and (self.integrator.lower() in ['rwm_then_thug', 'rwm_then_han']) and (not self.adaptiveδ):
            assert len(self.δ) == 2, "if δ is a list, must have length 2."
            for δ in self.δ:
                assert isinstance(δ, float) and δ > 0.0, "each δ in the list of δs must be a float and positive."
        else:
            raise ValueError("δ can only be a list when using rwm_then_thug or rwm_then_han integrator and adaptiveδ=False.")
        assert (self.εs_fixed is None) or all(x>y for x, y in zip(self.εs_fixed, self.εs_fixed[1:])), "εs must be a strictly decreasing list, or None."
        assert (self.prop_esjd > 0.0) and (self.prop_esjd <= 1.0), "self.prop_esjd must be in [0, 1]."
        assert (self.prop_hug >=0) and (self.prop_hug <= 1.0), "prop_hug must be in [0, 1]."
        assert self.δmin > 0.0, "δmin must be larger than 0."
        assert self.δmin <= self.δmax, "δmin must be less than or equal to δmax."
        assert self.εmin > 0.0, "εmin must be larger than 0."
        assert (self.Bmin > 1) and (self.Bmax > self.Bmin), "Bmin must be at least 1 and Bmax must be larger than Bmin."
        assert (self.min_pm >= 0.0) and (self.min_pm <= 1.0), "min_pm must be in [0, 1]."
        if isinstance(self.pm_target, float):
            assert (self.pm_target >= 0) and (self.pm_target <= 1.0), "pm_target must be in [0, 1]."
        elif isinstance(self.pm_target, list) and (self.integrator.lower() not in ['rwm_then_thug', 'rwm_then_han']):
            raise ValueError("pm_target must be float if integrator is not rwm_then_thug or rwm_then_han.")
        else: # is it a list and the integrator is rwm_then_thug
            for pmt in self.pm_target:
                assert (pmt >= 0) and (pmt <= 1.0), "each element in pm_target must be in [0, 1]."
        if isinstance(self.pd_target, float):
            assert (self.pd_target >= 0) and (self.pd_target <= 1.0), "pd_target must be in [0, 1]."
        elif isinstance(self.pd_target, list) and (self.integrator.lower() not in ['rwm_then_thug', 'rwm_then_han']):
            raise ValueError("pd_target must be float if integrator is not rwm_then_thug or rwm_then_han.")
        else:
            for pdt in self.pd_target:
                assert (pdt >= 0) and (pdt <= 1.0), "each element in pd_target must be in [0, 1]."
        if isinstance(self.dm_target, float):
            assert (self.dm_target >= 0) and (self.dm_target <= 1.0), "dm_target must be in [0, 1]."
        elif isinstance(self.md_target, list) and (self.integrator.lower() not in ['rwm_then_thug', 'rwm_then_han']):
            raise ValueError("dm_target must be float if integrator is not rwm_then_thug or rwm_then_han.")
        else:
            for mdt in self.md_target:
                assert (mdt >= 0) and (mdt <= 1.0), "each element in md_target must be in [0, 1]."
        assert (self.pm_switch >= 0) and (self.pm_switch <= 1.0), "pm_switch must be in [0, 1]."
        assert self.integrator.lower() in ['rwm', 'thug', 'rwm_then_thug', 'hug_and_nhug', 'rwm_then_han'], "integrator must be one of 'RWM', 'THUG', 'RWM_THEN_THUG', 'HUG_AND_HUG', 'RWM_THEN_HAN', 'RWM_KERNEL'."
        assert (self.εprop_switch >= 0.0) and (self.εprop_switch <= 1.0), "εprop_switch must be in [0, 1]."
        assert (self.ε0_manual is None) or (self.ε0_manual >= 0.0), "ε0_manual must be larger than 0 or must be None."
        if isinstance(self.quantile_value, float):
            assert (self.quantile_value >= 0) and (self.quantile_value <= 1.0), "quantile_value must be in [0, 1]."
        elif isinstance(self.quantile_value, list) and (self.integrator.lower() in ['rwm_then_thug', 'rwm_then_han']):
            assert len(self.quantile_value) == 2, "When quantile_value is a list, it must have two elements."
            for q in self.quantile_value:
                assert isinstance(q, float), "each quantile_value must be a float."
        else:
            raise ValueError("quantile_value must be float, or can be a list of length 2 with 2 float only when the integrator is rwm_then_thug or rwm_then_han.")
        assert self.proxy_ap_metric in ['pm', 'pd', 'dm'], "proxy_ap_metric must be one of 'pm', 'pd', or 'dm'."
        assert self.initialization in ['prior', 'manual'], "initialization must be one of 'prior' or 'manual'."
        assert self.switch_strategy in ['εprop', 'pm'], "switch_strategy must be one of 'εprop' or 'pm'."
        if isinstance(self.z0_manual, np.ndarray):
            if self.z0_manual.shape != (self.N, 2*self.d):
                raise ValueError("z0_manual must have shape (N, 2d).")
        assert self.stopping_criterion.issubset({'maxiter', 'εmin', 'pm'}), "stopping criterion must be a subset of maxiter, εmin and pm."
        assert self.resampling_scheme in ['multinomial', 'systematic'], "resampling scheme must be one of multinomial or resampling."
        assert self.projection_method in ['linear', 'qr']
        assert len(self.stopping_criterion) >= 1, "There must be at least one stopping criterion."
        assert (self.δadaptation_method in ['ap', 'esjd', 'esjd_and_pm']) or (self.δadaptation_method is None), "δadaptation_method must be None, 'ap', 'esjd', or 'esjd_and_pm'."
        if (self.δadaptation_method is None) and self.adaptiveδ:
            raise ValueError("When δadaptation_method is None, adaptiveδ must be set to False.")
        if (self.δadaptation_method is not None) and (not self.adaptiveδ):
            raise ValueError("When δadaptation_method is not None, adaptiveδ must be set to True.")
        if self.adaptiveB and self.δadaptation_method not in ['esjd_and_pm', None]:
            raise ValueError("When adaptiveB=True, δadaptation_method must be set to esjd_and_pm, or to None.")
        if self.adaptiveB and (not self.low_memory):
            raise ValueError("When B is adaptive, we need to use low memory as we cannot save particles into an array: they have a different size each time.")
        if (not self.adaptiveB) and (self.δadaptation_method == 'esjd_and_pm'):
            raise ValueError("When B is not adaptive, cannot choose δadaptation_method=esjd_and_pm.")
        if self.adaptiveB and (self.δadaptation_method is None) and self.adaptiveδ:
            raise ValueError("If you are trying to adapt only B, you need adaptiveδ=False and δadaptation_method=None.")
        # Can only use adaptiveN when adaptiveB=True, δadaptation_method=None and adaptiveδ=False
        if self.adaptiveN and ((not self.adaptiveB) or (self.δadaptation_method is not None) or (self.adaptiveδ)):
            raise ValueError("adaptiveN is only valid when adaptiveB=True, δadaptation_method=None, and adaptiveδ=False.")
        if self.adaptiveN and (not self.low_memory):
            raise ValueError("N cannot be adapted when low_memory=False.")

        # Create functions and variables based on input arguments
        self.verboseprint = print if self.verbose else lambda *a, **k: None  # Prints only when verbose is true
        self.univariate = True if (self.manifold.get_codimension() == 1) else False # basically keeps track if it is uni or multi variate.
        self.switched = False
        self.δs = [self._get_δ()]
        self.Bs = [self.B]
        self.Ns = [self.N]
        self.resampling_rng = default_rng(seed=self.resampling_seed)
        # Create a new variable B_size. This is to determine the sizes of arrays. basically
        # basically the idea is that this would be different whether it is METROPOLISED or not.
        # this variable would not be affected by later versions of the program that adapt B.
        self.Bsize = lambda: self.B if not self.metropolised else 1

        # Choose correct integrator to use
        if (self.integrator.lower() == 'rwm') or (self.integrator.lower() == 'rwm_then_thug') or (self.integrator.lower() == 'rwm_then_han'):
            if not self.metropolised:
                # Choose Random Walk Metropolis integrator
                self.verboseprint("Integrator: RWM.")
                self.ψ_generator = lambda B, δ: generate_RWMIntegrator(B, δ) # This is now a function that given B, δ it returns a function that integrates with those parameters
                self.ψ = self.ψ_generator(self.B, self._get_δ())
            else:
                # When metropolised, I just output the final point instead
                self.verboseprint("Integrator: RWM METROPOLISED.")
                self.ψ_generator = lambda B, δ: generate_RWMIntegrator(B, δ, metropolised=self.metropolised)
                self.ψ = self.ψ_generator(self.B, self._get_δ())
        elif self.integrator.lower() == 'thug':
            if not self.metropolised:
                self.verboseprint("Integrator: THUG.")
                # Instantiate the class, doesn't matter which ξ0 or logpi we use.
                THUGSampler = TangentialHugSampler(self.manifold.sample(advanced=True), self.B*self._get_δ(), self.B, self.N, 0.0, self.manifold.logprior, self.manifold.fullJacobian, method=self.projection_method, safe=True)
                self.ψ_generator = THUGSampler.generate_hug_integrator # again, this takes B, δ and returns an integrator (notice logpi doesn't matter)
                self.ψ = self.ψ_generator(self.B, self._get_δ())
            else:
                self.verboseprint("Integrator: THUG METROPOLISED.")
                THUGSampler = TangentialHugSampler(self.manifold.sample(advanced=True), self.B*self._get_δ(), self.B, self.N, 0.0, self.manifold.logprior, self.manifold.fullJacobian, method=self.projection_method, safe=True)
                self.ψ_generator = lambda B, δ: THUGSampler.generate_hug_integrator(B, δ, metropolised=self.metropolised) # again, this takes B, δ and returns an integrator (notice logpi doesn't matter)
                self.ψ = self.ψ_generator(self.B, self._get_δ())
        elif self.integrator.lower() == 'hug_and_nhug':
            if not self.metropolised:
                self.verboseprint("Integrator: HUG + NHUG.")
                self.verboseprint("Prop Hug  : ", self.prop_hug)
                # Instantiate the class, doesn't matter which ξ0 or logpi we use.
                THUGSampler = TangentialHugSampler(self.manifold.sample(advanced=True), self.B*self._get_δ(), self.B, self.N, 0.0, self.manifold.logprior, self.manifold.fullJacobian, method=self.projection_method, safe=True)
                self.ψ_generator = lambda B, δ: THUGSampler.generate_hug_and_nhug_integrator(B, δ, prop_hug=self.prop_hug) # again, this takes B, δ and returns an integrator (notice logpi doesn't matter)
                self.ψ = self.ψ_generator(self.B, self._get_δ())
            else:
                self.verboseprint("Integrator: HUG + NHUG METROPOLISED.")
                self.verboseprint("Prop Hug  : ", self.prop_hug)
                THUGSampler = TangentialHugSampler(self.manifold.sample(advanced=True), self.B*self._get_δ(), self.B, self.N, 0.0, self.manifold.logprior, self.manifold.fullJacobian, method=self.projection_method, safe=True)
                self.ψ_generator = lambda B, δ: THUGSampler.generate_hug_and_nhug_integrator(B, δ, prop_hug=self.prop_hug, metropolised=self.metropolised) # again, this takes B, δ and returns an integrator (notice logpi doesn't matter)
                self.ψ = self.ψ_generator(self.B, self._get_δ())
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
                # Since εs are provided, we need to fix maxiter to its length
                self.maxiter = len(self.εs)
            else:
                raise ValueError("Invalid initialization specifications.")
        else:
            raise ValueError("Invalid initialization specifications.")

        # Choose correct distance function
        if self.univariate:
            # Computes q(z) for each initial particle for a univariate problem Rn -> R
            self.compute_distances = self._compute_distances_univariate
            # Computes ||z_{n, k} - z_{n, 0}||^2 for all
        else:
            # Computes q(z) for each initial particle for a multivariate problem Rn -> Rm
            self.compute_distances = self._compute_distances_multivariate

        # Determine whether to switch RWM -> THUG or not
        if (self.integrator.lower() == 'rwm_then_thug') or (self.integrator.lower() == 'rwm_then_han'):
            self.switch = True
        else:
            self.switch = False

        # Choose correct stopping criterion based on user input
        stopping_criterion_string = ''  # for printing purposes
        if ('maxiter' in self.stopping_criterion) or (not self.adaptiveε):
            maximum_iterations = self.maxiter if self.adaptiveε else len(self.εs)-1
            self.check_iterations = lambda: self.n <= maximum_iterations
            stopping_criterion_string += 'maxiter, '
        else:
            self.check_iterations = lambda: True
        if 'εmin' in self.stopping_criterion:
            self.check_min_tolerance = lambda: (abs(self.εs[self.n-1]) >= self.εmin)
            stopping_criterion_string += 'εmin, '
        else:
            self.check_min_tolerance = lambda: True
        if 'pm' in self.stopping_criterion:
            self.check_pm = lambda: (self.PROP_MOVED[self.n-1] >= self.min_pm)
            stopping_criterion_string += 'pm.'
        else:
            self.check_pm = lambda: True
        self.check_stopping_criterion = lambda: self.check_iterations() and self.check_min_tolerance() and self.check_pm()
        self.verboseprint("Stopping criterion: ", stopping_criterion_string)

        # Choose resampling scheme
        if self.resampling_scheme == 'multinomial':
            resample = lambda W: self.resampling_rng.choice(a=arange(self.N*(self.Bsize()+1)), size=self.N, p=W.flatten())
            self.verboseprint("Resampling: MULTINOMIAL.")
        elif self.resampling_scheme == 'systematic':
            resample = lambda W: systematic(W.flatten(), self.N)
            self.verboseprint("Resampling: SYSTEMATIC.")
        else:
            raise ValueError("Resampling scheme must be either multinomial or systematic.")
        self._resample = resample

        # Choose how to initiate and update Wbar and ESJD_CHANG
        if self.adaptiveB:
            # if adaptive, we cannot use numpy arrays, we must use lists
            self.init_Wbar       = lambda: [zeros(self.N*(self.Bsize()+1))]
            self.init_ESJD_CHANG = lambda: [zeros(self.Bsize()+1)]
            # if adaptive, we need to append to the list
            self.update_Wbar       = self._update_Wbar_when_B_adaptive
            self.update_ESJD_CHANG = self._update_ESJD_CHANG_when_B_adaptive
        else:
            # if not adaptive, initiate as zero arrays
            self.init_Wbar       = lambda: zeros(self.N*(self.Bsize()+1))
            self.init_ESJD_CHANG = lambda: zeros(self.Bsize()+1)
            # if not adaptive, update by stacking on the arrays
            self.update_Wbar       = self._update_Wbar_when_B_not_adaptive
            self.update_ESJD_CHANG = self._update_ESJD_CHANG_when_B_not_adaptive

        # Initialize and update W_SMC
        if self.adaptiveN:
            # Initialize and update W_SMC storage
            self.init_W_SMC        = lambda: [zeros(self.N)]
            self.update_W_SMC      = self._update_W_SMC_when_N_adaptive
            # Initialize and update K_RESAMPLED storage
            self.init_K_RESAMPLED   = lambda: [zeros(self.N)]
            self.update_K_RESAMPLED = self._update_K_RESAMPLED_when_N_adaptive
            # Initialize and update N_RESAMPLED storage
            self.init_N_RESAMPLED   = lambda: [zeros(self.N)]
            self.update_N_RESAMPLED = self._update_N_RESAMPLED_when_N_adaptive
        else:
            # Initialize and update W_SMC storage
            self.init_W_SMC      = lambda: zeros(self.N)
            self.update_W_SMC      = self._update_W_SMC_when_N_not_adaptive
            # Initialize and update K_RESAMPLED storage
            self.init_K_RESAMPLED   = lambda: zeros(self.N)
            self.update_K_RESAMPLED = self._update_K_RESAMPLED_when_N_not_adaptive
            # Initialize and update N_RESAMPLED storage
            self.init_N_RESAMPLED   = lambda: zeros(self.N)
            self.update_N_RESAMPLED = self._update_N_RESAMPLED_when_N_not_adaptive

    def _update_Wbar_when_B_adaptive(self, W):
        """Updates Wbar when adaptiveB=True."""
        self.Wbar.append(W.flatten())

    def _update_Wbar_when_B_not_adaptive(self, W):
        """Updates Wbar when adaptiveB=False."""
        self.Wbar = vstack((self.Wbar, W.flatten()))

    def _update_ESJD_CHANG_when_B_adaptive(self, Z, W):
        """Updates ESJD_CHANG when adpativeB=True."""
        self.ESJD_CHANG.append(self._compute_esjd_br_chang(Z, W))

    def _update_ESJD_CHANG_when_B_not_adaptive(self, Z, W):
        """Updates ESJD_CHANG when adaptiveB=False."""
        self.ESJD_CHANG = vstack((self.ESJD_CHANG, self._compute_esjd_br_chang(Z, W)))

    def _update_W_SMC_when_N_adaptive(self, normalized_SMC_weights):
        """Updates W_SMC when adaptiveN=True."""
        self.W_SMC.append(normalized_SMC_weights)

    def _update_W_SMC_when_N_not_adaptive(self, unnormalized_SMC_weights):
        """Updates W_SMC when adaptiveN=False."""
        self.W_SMC = vstack((self.W_SMC, unnormalized_SMC_weights))

    def _update_K_RESAMPLED_when_N_adaptive(self, indices):
        """Updates when adaptiveN=True."""
        self.K_RESAMPLED.append(indices)

    def _update_K_RESAMPLED_when_N_not_adaptive(self, indices):
        """Updates when adaptiveN=False."""
        self.K_RESAMPLED = vstack((self.K_RESAMPLED, indices))

    def _update_N_RESAMPLED_when_N_adaptive(self, indices):
        """Updates when adaptiveN=True."""
        self.N_RESAMPLED.append(indices)

    def _update_N_RESAMPLED_when_N_not_adaptive(self, indices):
        """Updates when adaptiveN=False."""
        self.N_RESAMPLED = vstack((self.N_RESAMPLED, indices))

    def _set_B(self, B):
        """Sets the new value of B."""
        self.B = B
        self.Bs.append(B)

    def _set_δ(self, δ):
        """Sets new value of δ."""
        self.δ = δ

    def _get_δ(self):
        """Grabs δ. This is now needed because we are allowing self.δ to be a list of
        two δs when using rwm_then_thug. For that reason, self.δ could be a list and so
        whenever we call self.δ things would break. This function basically allows us to
        use the correct δ."""
        if isinstance(self.δ, list) and (not self.switched):
            return self.δ[0]
        elif isinstance(self.δ, list) and self.switched:
            return self.δ[1]
        elif isinstance(self.δ, float):
            return self.δ
        else:
            raise ValueError("Something went wrong when grabbing δ.")

    def _get_B(self):
        """Returns B"""
        return self.B

    def _get_quantile_value(self):
        """Grabs the quantile value in a similar way in which _get_δ() grabs δ."""
        if isinstance(self.quantile_value, list) and (not self.switched):
            return self.quantile_value[0]
        elif isinstance(self.quantile_value, list) and self.switched:
            return self.quantile_value[1]
        elif isinstance(self.quantile_value, float):
            return self.quantile_value
        else:
            raise ValueError("Couldnt get quantile value. ")

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
            self._update_distances(distances)
            # determine next ε as quantile of distances
            ε = min(self.εs[self.n-1], quantile(unique(distances), self._get_quantile_value()))
            self.εs.append(ε)
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηε, ε))

    def _update_distances(self, distances):
        """This is used to either append or overwrite distances based on whether
        the parameter low_memory is true."""
        if self.low_memory:
            self.DISTANCES = distances
        else:
            self.DISTANCES = vstack((self.DISTANCES, distances))

    def _compute_esjd_br_chang(self, Z, W):
        """Computes Expected Squared Jump Distance (Before Resampling) a la Chang.
        Chang's original metric was:

        ESJD-BR_{n, k} = sum_{i=1}^N bar{W}_{n, k}^{(i)} ||z_{n, k}^{(i)} - z_{n, 0}^{(i)}||^2

        But in our case I want to use the x-distances.

                    || x_{n, k}^{(i)} - x_{n, 0}^{(i)} ||^2
        It expects Z to be (N, B+1, 2d).
        """
        return np.sum(W * (norm(Z[:, :, :self.d] - Z[:, 0:1, :self.d], axis=2)**2), axis=0)

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

    def _output_adapted_δ(self):
        if self.proxy_ap_metric == 'pm':
            return clip(exp(log(self.δ) + 0.5*(self.PROP_MOVED[self.n] - self._get_pm_target())), self.δmin, self.δmax)
        elif self.proxy_ap_metric == 'pd':
            return clip(exp(log(self.δ) + 0.5*(self.P_DIVERSITY[self.n] - self._get_pd_target())), self.δmin, self.δmax)
        elif self.proxy_ap_metric == 'dm':
            return clip(exp(log(self.δ) + 0.5*(self.DIV_MOVED[self.n] - self._get_dm_target())), self.δmin, self.δmax)

    def _update_δ_B_ψ(self):
        """If self.adaptiveδ is True, then we adapt based on proxy AP or on ESJD. Otherwise
        we keep it the same. When adapting δ, we must remember to adapt ψ since
        now the integrator will be different."""
        if self.adaptiveδ:
            if self.δadaptation_method == 'ap':
                self.δ = self._output_adapted_δ()
            elif self.δadaptation_method == 'esjd':
                # Compute a value of B that avoids waisting resources
                B_efficient = np.argmax(np.cumsum(self.ESJD_CHANG[-1]) >= np.sum(self.ESJD_CHANG[-1])*self.prop_esjd)
                # Update delta to make sure we integrate the same amount
                self._set_δ(clip((B_efficient / self.B)*self.δ, self.δmin, self.δmax))
            elif self.δadaptation_method == 'esjd_and_pm' and self.adaptiveB:
                # if the average ESJD on the last 3 integration steps is above the median, then we increase B
                if np.mean(self.ESJD_CHANG[-1][-2:]) > np.median(self.ESJD_CHANG[-1]):
                    # # first find δ based on PM, but only use it to update B. Keep δ as is
                    # δ_new = clip(exp(log(self.δ) + 0.5*(self.PROP_MOVED[self.n] - self._get_pm_target())), self.δmin, self.δmax)
                    # # then update B based on that
                    # self._set_B(clip(int(self.B*δ_new/self.δ), self.Bmin, self.Bmax))
                    # self.verboseprint("\tIncreasing B={} and keeping δ={} fixed.".format(self.B, self.δ))
                    #### JUST INCREASE DELTA
                    self._set_δ(self._output_adapted_δ())
                else:
                    # in this case, first use ESJD to find Befficient, just like above
                    B_efficient = clip(np.argmax(np.cumsum(self.ESJD_CHANG[-1]) >= np.sum(self.ESJD_CHANG[-1])*self.prop_esjd), self.Bmin, self.Bmax)
                    T_efficient = B_efficient * self.δ
                    # then use PM to find step size
                    δ_efficient = self.B*self.δ/B_efficient
                    δ_new = self._output_adapted_δ()
                    self._set_δ(δ_new)
                    # divide T_efficient by new step size to find new B
                    self._set_B(clip(int(T_efficient / δ_new), self.Bmin, self.Bmax))
            else:
                raise ValueError("δadaptation_method is neither 'ap' nor 'esjd' but adaptiveδ is True.")
            # Update delta and store it
            self.δs.append(self.δ)
            self.ψ = self.ψ_generator(self.B, self.δ)
            self.verboseprint("\tStep-size adapted to: {:.16f} with strategy {}".format(self.δ, self.δadaptation_method))
            if self.δadaptation_method == 'esjd_and_pm' and self.adaptiveB:
                self.verboseprint("\tNumber of steps adapted to: {}".format(self.B))
        elif self.adaptiveB and (not self.adaptiveδ) and (self.δadaptation_method is None):
            # In this case we only adapt B
            self._set_B(clip(np.argmax(np.cumsum(self.ESJD_CHANG[-1]) >= np.sum(self.ESJD_CHANG[-1])*self.prop_esjd), self.Bmin, self.Bmax))
            self.ψ = self.ψ_generator(self.B, self.δ)
            self.verboseprint("\tNumber of steps adapted to: {}".format(self.B))
            if self.adaptiveN:
                # adapt N based on budget
                self.N = int(self.NBbudget / self.B)
                self.Ns.append(self.N)
                self.verboseprint("\tNumber of particles adapted to: {}".format(self.N))
        else:
            self.δs.append(self._get_δ())
            self.verboseprint("\tStep-size kept fixed at: {:.16f}".format(self._get_δ()))

    def switch_integrator(self):
        """Switches from RWM to THUG, or from RWM to HAN."""
        # the next 3 lines are taken verbatim from __init__ when integrator = 'THUG'
        x0 = self.manifold.sample(advanced=True)
        self.sampled_x0 = x0
        THUGSampler = TangentialHugSampler(x0, self.B*self._get_δ(), self.B, self.N, 0.0, self.manifold.logprior, self.manifold.fullJacobian, method=self.projection_method, safe=True)
        if self.integrator.lower() == 'rwm_then_thug':
            self.ψ_generator = lambda B, δ: THUGSampler.generate_hug_integrator(B, δ, metropolised=self.metropolised)# again, this takes B, δ and returns an integrator (notice logpi doesn't matter)
        elif self.integrator.lower() == 'rwm_then_han':
            self.ψ_generator = lambda B, δ: THUGSampler.generate_hug_and_nhug_integrator(B, δ, prop_hug=self.prop_hug, metropolised=self.metropolised)
        else:
            raise NotImplementedError("Attempted switching integrator even though it is neither rwm_then_thug nor rwm_then_han.")
        self.ψ = self.ψ_generator(self.B, self._get_δ())
        # Store when the switch happend
        self.n_switch = self.n  # store when the switch happens
        self.switched = True
        self.verboseprint("\n")
        self.verboseprint("####################################")
        if self.integrator.lower() == 'rwm_then_thug':
            if not self.metropolised:
                self.verboseprint("### SWITCHING TO THUG INTEGRATOR ###")
            else:
                self.verboseprint("### SWITCHING TO THUG INTEGRATOR METROPOLISED ###")
        else:
            if not self.metropolised:
                self.verboseprint("### SWITCHING TO HAN INTEGRATOR ###")
            else:
                self.verboseprint("### SWITCHING TO HAN INTEGRATOR METROPOLISED ###")
        self.verboseprint("####################################")
        self.verboseprint("\n")

    def initialize_particles(self):
        """Initializes the particles and stores them in a separate variable, for checking purposes."""
        z0 = self.initializer()
        self.starting_particles = z0
        return z0

    def _get_pm_target(self):
        """Returns the correct pm target both if it is a float or a list."""
        if isinstance(self.pm_target, float):
            return self.pm_target
        elif isinstance(self.pm_target, list):
            if self.switched:
                return self.pm_target[1]
            else:
                return self.pm_target[0]
        else:
            raise ValueError("pm target must be list or float, but found: ", type(self.pm_target))

    def _get_pd_target(self):
        """Returns the correct pd target both if it is a float or a list."""
        if isinstance(self.pd_target, float):
            return self.pd_target
        elif isinstance(self.pd_target, list):
            if self.switched:
                return self.pd_target[1]
            else:
                return self.pd_target[0]
        else:
            raise ValueError("pd target must be list or float, but found: ", type(self.pd_target))

    def _get_dm_target(self):
        """Returns the correct md target both if it is a float or a list."""
        if isinstance(self.dm_target, float):
            return self.dm_target
        elif isinstance(self.dm_target, list):
            if self.switched:
                return self.dm_target[1]
            else:
                return self.dm_target[0]
        else:
            raise ValueError("dm target must be list or float, but found: ", type(self.dm_target))

    def _resample(self, W):
        """Returns resampled indeces."""
        raise NotImplementedError("Resampling not implemented.")

    def _whittle_down_to_new_N(self, z):
        """This is used when adaptiveN=True. In this case, we randomly pick self.N
        particles to continue the algorithm."""
        if self.adaptiveN:
            # Choose indeces at random (uniformly)
            N_old = z.shape[0]
            indices = choice(a=arange(N_old).astype(int), size=self.N, p=np.repeat(1/N_old, repeats=N_old)).astype(int)
            return z[indices, :]
        else:
            return z

    def sample(self):
        """Samples using the Markov Snippets algorithm."""
        start_time = time()
        #### STORAGE
        self.ZN          = zeros((1, self.N, 2*self.d))            # z_n^{(i)}
        self.ZNK         = zeros((1, self.N*(self.Bsize()+1), 2*self.d)) # z_{n, k}^{(i)} all the N(T+1) particles
        self.Wbar        = self.init_Wbar() #zeros(self.N*(self.Bsize()+1))
        self.W_SMC       = self.init_W_SMC()                       # weights for underlying SMC sampler
        self.DISTANCES   = zeros(self.N)                      # distances are computed on the z_n^{(i)}
        self.ESS         = [self.N*(self.Bsize()+1)]                     # ESS computed on Wbar so in reference to all N(T+1) particles
        self.ESS_SMC     = [self.N]                         # ESS of the underlying SMC sampler
        self.K_RESAMPLED = self.init_K_RESAMPLED() #zeros(self.N)                    # Stores indeces within the trajectory that have been resampled
        self.N_RESAMPLED = self.init_N_RESAMPLED() #zeros(self.N)                    # Stores the indeces of the particle-trajectory that have been resampled
        self.PROP_MOVED  = [1.0]                            # Stores proportion of particles moved forward on the trajectories
        self.P_DIVERSITY = [1.0]                            # particle_diversity is the equivalent of PM for the particle index rather than trajectory index.
        self.DIV_MOVED   = [1.0]                            # diversity_moved is the multiplication of prop_moved and p_diversity
        self.ESJD_CHANG  = self.init_ESJD_CHANG() #zeros(self.Bsize()+1)              # Expected Squared Jump Distance computed for each index k, a la Chang
        #### INITIALIZATION
        z = self.initialize_particles()   # (N, 2d)
        self.ZN[0] = z
        if self.initialization == 'prior':
            distances = self.compute_distances(z[:, :self.d])  # compute εmax and log_ηεmax and add them to the storage lists
            self.εmax = np.max(distances)
            self.εs.append(self.εmax)
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηε, self.εmax))
            self.verboseprint("Setting initial epsilon to εmax = {:.16f}".format(self.εmax))
        if self.initialization == 'manual':
            # when using manual initialization we need to set the first epsilon to the one provided
            # since εs = [ε0, ε1, ... εP] has size P+1, where P is the number of epochs.
            pass
        # Keep running until stopping criterion is met
        # In this case we stop if we reach the number of maximum iterations, or
        # if out ε becomes smaller than εmin or if we move less than self.min_pm particles
        self.n = 1
        try:
            while self.check_stopping_criterion(): #(self.n <= self.maxiter) and (abs(self.εs[self.n-1]) >= self.εmin) and (self.PROP_MOVED[self.n-1] >= self.min_pm):
                self.verboseprint("Iteration: ", self.n)
                self.verboseprint("\tQuantile Value: ", self._get_quantile_value())

                #### COMPUTE TRAJECTORIES
                Z = apply_along_axis(self.ψ, 1, z)                                     # (N, B+1, 2d)
                if not self.low_memory:
                    self.ZNK = vstack((self.ZNK, Z.reshape(1, self.N*(self.Bsize()+1), 2*self.d)))  # (n+1, N(B+1), 2d)
                self.verboseprint("\tTrajectories constructed.")

                #### DETERMINE TOLERANCE TO TARGET AT THIS ITERATION
                self._compute_nth_tolerance(z) # If adaptive, computes εn and logηεn, otherwise does nothing (already available when schedule is fixed)
                self.verboseprint("\tEpsilon: {:.16f}".format(self.εs[self.n]))

                #### COMPUTE WEIGHTS
                # Log-Denominator: shared for each point in the same trajectory
                log_μnm1_z  = apply_along_axis(self.log_ηs[self.n-1], 1, Z[:, 0, :self.d])         # (N, )
                log_μnm1_z  = repeat(log_μnm1_z, self.Bsize()+1, axis=0).reshape(self.N, self.Bsize()+1) # (N, B+1)
                # Log-Numerator: different for each point on a trajectory.
                log_μn_ψk_z = apply_along_axis(self.log_ηs[self.n], 2, Z[:, :, :self.d])         # (N, B+1)
                W = self._compute_weights(log_μnm1_z, log_μn_ψk_z)
                self.verboseprint("\tWeights computed and normalized.")
                # Store weights and ESS
                self.update_Wbar(W) #self.Wbar = vstack((self.Wbar, W.flatten()))
                self.ESS.append(1 / np.sum(W**2))   ### This only works because weights are already normalized!!!
                # Compute the ESS of the underlying SMC sampler. To do so, we need to compute the
                # underlying SMC weights. These are the \bar{w}_n. There is one of them for each particle
                # therefore we will have N weights, one per particle
                unnormalized_SMC_weights = np.mean(W, axis=1)  # these are unnormalized
                normalized_SMC_weights = unnormalized_SMC_weights / unnormalized_SMC_weights.sum() # these are normalized!
                self.update_W_SMC(normalized_SMC_weights) #vstack((self.W_SMC, normalized_SMC_weights))
                self.verboseprint("\tSMC Weights computed and normalized.")
                self.ESS_SMC.append(1 / np.sum(self.W_SMC[-1]**2))
                # Compute ESJD-BR (Chang's version)
                self.update_ESJD_CHANG(Z, W) # self.ESJD_CHANG = vstack((self.ESJD_CHANG, self._compute_esjd_br_chang(Z, W)))
                #### RESAMPLING
                resampling_indeces = self._resample(W)
                unravelled_indeces = unravel_index(resampling_indeces, (self.N, self.Bsize()+1))
                self.update_K_RESAMPLED(unravelled_indeces[1]) #vstack((self.K_RESAMPLED, unravelled_indeces[1]))
                self.update_N_RESAMPLED(unravelled_indeces[0]) #vstack((self.N_RESAMPLED, unravelled_indeces[0]))
                indeces = dstack(unravelled_indeces).squeeze()
                z = vstack([Z[tuple(ix)] for ix in indeces])     # (N, 2d)
                self.verboseprint("\tParticles Resampled.")

                #### REJUVENATE VELOCITIES
                z[:, self.d:] = normal(loc=0.0, scale=1.0, size=(self.N, self.d))
                if not self.low_memory:
                    self.ZN = vstack((self.ZN, z[None, ...]))
                self.verboseprint("\tVelocities refreshed.")

                #### ADAPT STEP SIZE
                # Compute proxy acceptance probability
                self.PROP_MOVED.append(sum(self.K_RESAMPLED[-1] >= 1) / self.N)
                self.verboseprint("\tProp Moved: {:.16f}".format(self.PROP_MOVED[self.n]))
                # Compute proxy particle diversity
                self.P_DIVERSITY.append(len(np.unique(self.N_RESAMPLED[-1])) / self.N)
                self.verboseprint("\tParticle Diversity: {:.16f}".format(self.P_DIVERSITY[self.n]))
                # Compute diversity moved
                self.DIV_MOVED.append(self.PROP_MOVED[-1] * self.P_DIVERSITY[-1])
                self.verboseprint("\tMoved Diversity: {:.16f}".format(self.DIV_MOVED[self.n]))
                # Adapt δ basedn on proxy acceptance probability
                self._update_δ_B_ψ()
                z = self._whittle_down_to_new_N(z)

                #### CHECK IF IT'S TIME TO SWITCH INTEGRATOR
                if (self.integrator.lower() == 'rwm_then_thug') or (self.integrator.lower() == 'rwm_then_han'):  # only happens when we allow switching
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
            self.total_time = time() - start_time
            print("ValueError was raised: ", e)
            return z
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
        self.mh_kernel_seed = SETTINGS['mh_kernel_seed'] # seed for the MH kernel
        self.switch_strategy = SETTINGS['switch_strategy'] # strategy used to determine when to switch RWM->THUG.
        self.resampling_seed = SETTINGS['resampling_seed'] # seed for resampling
        self.resampling_scheme = SETTINGS['resampling_scheme'] # resampling scheme to use
        self.stopping_criterion = SETTINGS['stopping_criterion'] # determines strategy used to terminate the algorithm

        # Check arguments types
        assert isinstance(self.N,  int), "N must be an integer."
        assert isinstance(self.B, int), "B must be an integer."
        assert isinstance(self.δ, float) or isinstance(self.δ, list), "δ must be a float or a list."
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
        assert isinstance(self.pm_target, float) or isinstance(self.pm_target, list), "pm_target must be float or list."
        assert isinstance(self.pm_switch, float), "pm_switch must be float."
        assert isinstance(self.prior_seed, int), "prior_seed must be integer."
        assert isinstance(self.low_memory, bool), "low_memory must be bool."
        assert isinstance(self.integrator, str), "integrator must be a string."
        assert isinstance(self.εprop_switch, float), "εprop_switch must be a float."
        assert (self.ε0_manual is None) or isinstance(self.ε0_manual, float), "ε0_manual must be float or None."
        assert isinstance(self.quantile_value, float) or isinstance(self.quantile_value, list), "quantile_value must be a float or list."
        assert isinstance(self.initialization, str), "initialization must be a string."
        assert isinstance(self.mh_kernel_seed, int), "mh_kernel_seed must be an integer."
        assert isinstance(self.switch_strategy, str), "switch_strategy must be a string."
        assert isinstance(self.resampling_seed, int), "resampling_seed must be integer."
        assert isinstance(self.resampling_scheme, str), "resampling_scheme must be a string."
        assert isinstance(self.stopping_criterion, set), "stopping criterion must be a set."

        # Check argument values
        if isinstance(self.δ, float):
            assert self.δ > 0.0, "δ must be larger than 0."
        elif isinstance(self.δ, list) and (self.integrator.lower() in 'rwm_then_thug') and (not self.adaptiveδ):
            assert len(self.δ) == 2, "if δ is a list, must have length 2."
            for δ in self.δ:
                assert isinstance(δ, float) and δ > 0.0, "each δ in the list of δs must be a float and positive."
        else:
            raise ValueError("δ can only be a list when using rwm_then_thug integratora and adaptiveδ=False.")
        assert (self.εs_fixed is None) or all(x>y for x, y in zip(self.εs_fixed, self.εs_fixed[1:])), "εs must be a strictly decreasing list, or None."
        assert self.δmin > 0.0, "δmin must be larger than 0."
        assert self.δmin <= self.δmax, "δmin must be less than or equal to δmax."
        assert self.εmin > 0.0, "εmin must be larger than 0."
        assert (self.min_pm >= 0.0) and (self.min_pm <= 1.0), "min_pm must be in [0, 1]."
        if isinstance(self.pm_target, float):
            assert (self.pm_target >= 0) and (self.pm_target <= 1.0), "pm_target must be in [0, 1]."
        elif isinstance(self.pm_target, list) and (self.integrator.lower() != 'rwm_then_thug'):
            raise ValueError("pm_target must be float if integrator is not rwm_then_thug.")
        else: # is it a list and the integrator is rwm_then_thug
            for pmt in self.pm_target:
                assert (pmt >= 0) and (pmt <= 1.0), "each element in pm_target must be in [0, 1]."
        assert self.integrator.lower() in ['rwm', 'thug', 'rwm_then_thug'], "integrator must be one of 'RWM', 'THUG', or 'RWM_THEN_THUG'."
        assert (self.εprop_switch >= 0.0) and (self.εprop_switch <= 1.0), "εprop_switch must be in [0, 1]."
        assert (self.ε0_manual is None) or (self.ε0_manual >= 0.0), "ε0_manual must be larger than 0 or must be None."
        if isinstance(self.quantile_value, float):
            assert (self.quantile_value >= 0) and (self.quantile_value <= 1.0), "quantile_value must be in [0, 1]."
        elif isinstance(self.quantile_value, list) and (self.integrator.lower() == 'rwm_then_thug'):
            assert len(self.quantile_value) == 2, "When quantile_value is a list, it must have two elements."
            for q in self.quantile_value:
                assert isinstance(q, float), "each quantile_value must be a float."
        else:
            raise ValueError("quantile_value must be float, or can be a list of length 2 with 2 float only when the integrator is rwm_then_thug.")
        assert self.initialization in ['prior', 'manual'], "initialization must be one of 'prior' or 'manual'."
        assert self.switch_strategy in ['εprop', 'pm'], "switch_strategy must be one of 'εprop' or 'pm'."
        if isinstance(self.z0_manual, np.ndarray):
            if self.z0_manual.shape != (self.N, 2*self.d):
                raise ValueError("z0_manual must have shape (N, d).")
            # This is an SMC sampler, meaning the particles consist only of the position, not of the velocity.
            # Create a new variable, unique to SMC samplers, that stores the initial positions
            self.x0_manual = self.z0_manual[:, :self.d]
        assert self.stopping_criterion.issubset({'maxiter', 'εmin', 'pm'}), "stopping criterion must be a subset of maxiter, εmin and pm."
        assert self.resampling_scheme in ['multinomial', 'systematic'], "resampling scheme must be one of multinomial or resampling."
        assert len(self.stopping_criterion) >= 1, "There must be at least one stopping criterion."


        # Create functions and variables based on input arguments
        self.verboseprint = print if self.verbose else lambda *a, **k: None  # Prints only when verbose is true
        self.univariate = True if (self.manifold.get_codimension() == 1) else False # basically keeps track if it is uni or multi variate.
        self.switched = False
        self.δs = [self._get_δ()]
        self.resampling_rng = default_rng(seed=self.resampling_seed)
        # Generate seeds for each particle
        self.mh_rng   = default_rng(seed=self.mh_kernel_seed)
        self.mh_seeds = [self.mh_rng.integers(low=1000, high=9999) for _ in range(self.N)]

        # Choose correct KERNEL to use
        if (self.integrator.lower() == 'rwm') or (self.integrator.lower() == 'rwm_then_thug'):
            # Choose Random Walk Metropolis kernel
            self.verboseprint("Stochastic Kernel: RWM.")
            self.MH_kernel = lambda x, B, δ, log_ηε, seed: RWM(x, B*δ, 1, log_ηε, seed=seed)[0].flatten()
        elif self.integrator.lower() == 'thug':
            self.verboseprint("Stochastic Kernel: THUG.")
            # Instantiate the class, doesn't matter which ξ0 or logpi we use.
            self.THUGSampler = TangentialHugSampler(self.manifold.sample(advanced=True), self.B*self._get_δ(), self.B, self.N, 0.0, self.manifold.logprior, self.manifold.fullJacobian, method='linear', safe=True, seed=self.mh_kernel_seed)
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

        # Choose correct stopping criterion based on user input
        stopping_criterion_string = ''  # for printing purposes
        if 'maxiter' in self.stopping_criterion:
            self.check_iterations = lambda: self.n <= self.maxiter
            stopping_criterion_string += 'maxiter, '
        else:
            self.check_iterations = lambda: True
        if 'εmin' in self.stopping_criterion:
            self.check_min_tolerance = lambda: (abs(self.εs[self.n-1]) >= self.εmin)
            stopping_criterion_string += 'εmin, '
        else:
            self.check_min_tolerance = lambda: True
        if 'pm' in self.stopping_criterion:
            self.check_pm = lambda: (self.APS[self.n-1] >= self.min_pm)
            stopping_criterion_string += 'pm.'
        else:
            self.check_pm = lambda: True
        self.check_stopping_criterion = lambda: self.check_iterations() and self.check_min_tolerance() and self.check_pm()
        self.verboseprint("Stopping criterion: ", stopping_criterion_string)

        # Choose resampling scheme
        if self.resampling_scheme == 'multinomial':
            resample = lambda W: self.resampling_rng.choice(a=arange(self.N), size=self.N, p=W.flatten())
            self.verboseprint("Resampling: MULTINOMIAL.")
        elif self.resampling_scheme == 'systematic':
            resample = lambda W: systematic(W.flatten(), self.N)
            self.verboseprint("Resampling: SYSTEMATIC.")
        else:
            raise ValueError("Resampling scheme must be either multinomial or systematic.")
        self._resample = resample

    def _get_δ(self):
        """Grabs δ. This is now needed because we are allowing self.δ to be a list of
        two δs when using rwm_then_thug. For that reason, self.δ could be a list and so
        whenever we call self.δ things would break. This function basically allows us to
        use the correct δ."""
        if isinstance(self.δ, list) and (not self.switched):
            return self.δ[0]
        elif isinstance(self.δ, list) and self.switched:
            return self.δ[1]
        elif isinstance(self.δ, float):
            return self.δ
        else:
            raise ValueError("Something went wrong when grabbing δ.")

    def _get_quantile_value(self):
        """Grabs the quantile value in a similar way in which _get_δ() grabs δ."""
        if isinstance(self.quantile_value, list) and (not self.switched):
            return self.quantile_value[0]
        elif isinstance(self.quantile_value, list) and self.switched:
            return self.quantile_value[1]
        elif isinstance(self.quantile_value, float):
            return self.quantile_value
        else:
            raise ValueError("Couldnt get quantile value. ")

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
            self._compute_distances(distances)
            # determine next ε as quantile of distances
            # do not use clip because otherwise we will never finish
            ε = min(self.εs[self.n-1], quantile(unique(distances), self._get_quantile_value()))
            self.εs.append(ε)
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηε, ε))

    def _compute_distances(self, distances):
        """Determines whether to overwrite or append distances based on low_memory
        parameter."""
        if self.low_memory:
            self.DISTANCES = distances
        else:
            self.DISTANCES = vstack((self.DISTANCES, distances))

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
            self.δ = clip(exp(log(self.δ) + 0.5*(self.APS[self.n] - self._get_pm_target())), self.δmin, self.δmax)
            self.δs.append(self.δ)
            self.verboseprint("\tStep-size adapted to: {:.16f}".format(self.δ))
        else:
            self.δs.append(self._get_δ())
            self.verboseprint("\tStep-size kept fixed at: {:.16f}".format(self._get_δ()))

    def switch_kernel(self):
        """Switches from RWM to THUG kernels."""
        # the next 3 lines are taken verbatim from __init__ when integrator = 'THUG'
        # in the class initialization T, B, N, α, and logpi don't matter. Only thing that
        # matters is 'safe', 'fullJacobian', and 'method'.
        THUGSampler = TangentialHugSampler(self.manifold.sample(advanced=True), self.B*self._get_δ(), self.B, self.N, 0.0, self.manifold.logprior, self.manifold.fullJacobian, method='linear', safe=True, seed=self.mh_kernel_seed)
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

    def _get_pm_target(self):
        """Returns the correct pm target both if it is a float or a list."""
        if isinstance(self.pm_target, float):
            return self.pm_target
        elif isinstance(self.pm_target, list):
            if self.switched:
                return self.pm_target[1]
            else:
                return self.pm_target[0]
        else:
            raise ValueError("pm target must be list or float, but found: ", type(self.pm_target))

    def _resample(self, W):
        """Returns resampled indeces."""
        raise NotImplementedError("Resampling not implemented.")

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
        self.δs = [self._get_δ()]                     # Step sizes

        # If prior initialization, find εmax to start with
        if self.initialization == 'prior':
            self.εmax = np.max(self.DISTANCES)
            self.εs.append(self.εmax)
            self.log_ηs.append(FilamentaryDistribution(self.manifold.generate_logηε, self.εmax))
            self.verboseprint("Setting initial epsilon to εmax = {:.16f}".format(self.εmax))

        self.n = 1
        try:
            while self.check_stopping_criterion(): #(self.n <= self.maxiter) and (abs(self.εs[self.n-1]) >= self.εmin) and (self.APS[self.n-1] >= self.min_pm):
                self.verboseprint("Iteration: ", self.n)
                self.verboseprint("\tQuantile Value: ", self._get_quantile_value())

                # RESAMPLE PARTICLES
                if self.n == 1:
                    indeces = self._resample(self.WEIGHTS)
                    #self.resampling_rng.choice(a=arange(self.N), size=self.N, p=self.WEIGHTS)
                else:
                    indeces = self._resample(self.WEIGHTS[self.n-1])
                    #self.resampling_rng.choice(a=arange(self.N), size=self.N, p=self.WEIGHTS[self.n-1])
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
                M = lambda z, ix: self.MH_kernel(z, self.B, self._get_δ(), self.log_ηs[self.n], self.mh_seeds[ix])
                alive         = self.WEIGHTS[self.n] > 0.0
                alive_indeces = where(alive)[0]
                z_new         = deepcopy(z)
                for ix in alive_indeces:
                    z_new[ix] = M(z[ix], ix)
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

### SYSTEMATIC RESAMPLING (CHOPIN BOOK ON SMC)
def inverse_cdf(su, W):
    """Inverse CDF algorithm for a finite distribution.
        Parameters
        ----------
        su: (M,) ndarray
            M sorted uniform variates (i.e. M ordered points in [0,1]).
        W: (N,) ndarray
            a vector of N normalized weights (>=0 and sum to one)
        Returns
        -------
        A: (M,) ndarray
            a vector of M indices in range 0, ..., N-1
    """
    j = 0
    s = W[0]
    M = su.shape[0]
    A = np.empty(M, dtype=np.int64)
    for n in range(M):
        while su[n] > s:
            j += 1
            s += W[j]
        A[n] = j
    return A

def systematic(W, M):
    """Systematic resampling.
    """
    su = (rand(1) + np.arange(M)) / M
    return inverse_cdf(su, W)


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
