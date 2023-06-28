"""
This script will implement Markov Snippets targeting the G and K distribution
and it is parallelizable. Hopefully it should work. For the moment, we use a RWM
integrator. Taken from `markov_snippets.py`
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
from multiprocessing import Pool
from itertools import product
