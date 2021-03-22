import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Manifolds.Sphere import Sphere
from Manifolds.ManifoldAD import Manifold
from zappa import zappa_sampling, project
from utils import logf, logp, quick_3d_scatter, normalize
import time


def f(xyz):
    return np.sum((xyz)**2) - .25

MAD=Manifold(1,2,f)  #Instantiate with manifold autodiff

# Settings
mu = np.array([0, 0, 0])    # Center of the Sphere
r = 0.5                     # Radius

# Instantiate sphere and choose starting point on it
sphere = Sphere(mu, r)
x = sphere.to_cartesian([3.5, 3.5])

####TRACK TIME USING GRADIENTS IN CLOSED FORM###
start_time = time.time()   
# Run Zappa algorithm
samples = zappa_sampling(x, sphere, logf, logp, n=10000, sigma=0.5, tol=1.48e-08 , a_guess=1.0)

print("--- %s seconds for sampling with closed form gradient ---" % (time.time() - start_time))


####TRACK TIME USING GRADIENTS COMPUTED NUMERICALLY###

start_time = time.time()

# Run Zappa algorithm
samples = zappa_sampling(x, MAD, logf, logp, n=10000, sigma=0.5, tol=1.48e-08 , a_guess=1.0)

print("--- %s seconds for sampling using autodiff---" % (time.time() - start_time))
quick_3d_scatter(samples)