import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(r'/Users/ys18223/Documents/GitHub/ManifoldSampling')  ##Check your path
from Manifolds.ManifoldAD import Manifold
from zappa import zappa_sampling

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from numpy import pi
from scipy.stats import multivariate_normal, norm
import matplotlib.cm as cm
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
from scipy.optimize import root
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import qr, svd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils import logf, logp, quick_3d_scatter, normalize
from Manifolds.Sphere import Sphere
from zappa import zappa_sampling, project
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