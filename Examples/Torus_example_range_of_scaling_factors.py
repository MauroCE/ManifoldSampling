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
from joblib import Parallel, delayed
import pandas as pd


R = 1                            # Toroidal radius
r = 0.5                          # Polar radius
n = int(1e6)                     # Number of samples
m = 1                            # Number of constraints
d = 2                            # Dimension of manifold
mu = np.array([0.0, 0.0, 0.0])   # Center of the torus
θ= 3.5                      # θ for initial point on the sphere
ϕ= 3.5                        # ϕ for initial point on the sphere
sigma_range= [0.01,0.1,0.5,1,2,5,10]         # σ scaling for proposal (range)
tol = 1.48e-08                   # tolerance for root finding algorithm
a_guess = 1                      # initial guess for root-finding algorithm
maxiter = 10
use_their_project = False


###  Helper functions ###
# TORUS
def to_cartesian(theta_phi):
    """[θ, ϕ] --> [x, y, z]"""
    theta, phi = theta_phi[0], theta_phi[1]
    x = mu[0] + (R + r * np.cos(theta)) * np.cos(phi)
    y = mu[1] + (R + r * np.cos(theta)) * np.sin(phi)
    z = mu[2] + r * np.sin(theta)
    return np.array([x, y, z])

# TORUS
def grad(xyz):
    """Q"""
    xc, yc, zc = xyz - mu
    return np.array([
        2 * xc * (np.sqrt(xc**2 + yc**2) - R) / np.sqrt(xc**2 + yc**2),
        2 * yc * (np.sqrt(xc**2 + yc**2) - R) / np.sqrt(xc**2 + yc**2),
        2 * zc
    ])

# ANY
def tangent_basis(Q):
    """Given the Q matrix (the gradient) computes a basis for the tagent space it is normal to via SVD"""
    U = svd(Q.reshape(-1, m))[0]
    return U[:, m:]   # contains 2 columns. Each column is a basis

# ANY
def normalize(x):
    """Returns x as a unit vector"""
    return x / np.sqrt(np.sum(x**2))

# ANY
def project(x, v, Q, tol=None, a_guess=1, maxiter=maxiter):
    """Finds a such that q(x + v + a*Q) = 0"""
    opt_output = root(lambda a: q(x + v + a*Q), a_guess, tol=tol)
    return (opt_output.x, opt_output.success) # output flag for accept/reject

# ANY
def logf(x):
    """Implements the target distribution on the manifold"""
    # In our case, it's the uniform distribution
    return 0

# ANY
def log_proposal(v,σ):
    """Computes log density of the proposal. Right now its a gaussian"""
    return multivariate_normal.logpdf(v, mean=np.zeros(d), cov=(σ*2)*np.eye(d))

# TORUS
def q(xyz):
    """Constraint function for the sphere"""
    xc, yc, zc = xyz - mu   # center
    return (np.sqrt(xc**2 + yc**2) - R)**2 + zc**2 - r**2

def jacobian_newton(x, v, a, Q):
    """Jacobian for newton's method"""
    return np.dot(grad(x + v + a*Q), Q)

def project2(x, v, Q, tol=tol, a_guess=1, maxiter=maxiter):
    """Project, but their version"""
    a = 0
    i = 0
    flag = 1
    while np.sqrt(np.sum(q(x + v + a*Q)**2)) > tol:
        delta_a = jacobian_newton(x,v,a,Q) / -q(x+v+a*Q)
        a += delta_a
        i += 1
        if i > maxiter:
            flag = 0
            return a, flag
    return a, flag    

if use_their_project:
    project = project2

   

def torus_sampler(n,R,r,θ,ϕ,σ,tol):
    # Initial point on the sphere
    x = to_cartesian(np.array([theta, phi]))

    # Store the n sampes on the manifold
    samples = np.zeros((n, d+m))
    samples[0, :] = x
    aux_variable=np.empty((n,2))
    aux_variable[:,1]=σ
    i = 1

    # Sample log-uniform variates for MH accept/reject step
    logu = np.log(np.random.rand(n))

    while i < n:
        
        # Compute gradient and bases at x
        Qx = grad(x)                 # Gradient at x (3,), 
        gx_basis = normalize(Qx)     # ON basis for gradient (3,)
        tx_basis = tangent_basis(Qx) # ON basis for tangent space at x using SVD (3, 2)
        
        # Sample along tangent
        v_sample = σ*randn(d) # MVN with scaling σ
        v = tx_basis @ v_sample   # Multiply each basis vector with each MVN sample
        
        # Forward Projection
        a, flag = project(x, v, Qx, tol, a_guess)
        if flag == 0:              # Projection failed
            samples[i, :] = x      # Stay at x
            i += 1
            aux_variable[i-1,0]=2 #rejection from the first projection step
            continue
        y = x + v + a*Qx           # Compute projected point
        
        # Compute v' and w' from y
        Qy = grad(y)                        # Gradient at y (3, )
        gy_basis = normalize(Qy)            # ON basis for gradient (3, )
        ty_basis = tangent_basis(Qy)        # ON basis for tangent space at y using SVD (3, 2)
        v_prime_sample = (x - y) @ ty_basis # Components along tangent
            
        # Metropolis-Hastings
        if logu[i] > logf(y) + log_proposal(v_prime_sample,σ) - logf(x) - log_proposal(v_sample,σ):
            samples[i, :] = x      # Reject, stay at x
            aux_variable[i-1,0]=1 #MH rejection
            i += 1
            continue
        
        # Backward Projection
        v_prime = v_prime_sample @ ty_basis.T
        a_prime, flag = project(y, v_prime, Qy, tol, a_guess)
        if flag == 0:
            samples[i, :] = x     # projection failed, stay at x
            i += 1
            aux_variable[i-1,0]=3 #rejection from the second step
            continue
            
        # Accept move
        x = y
        samples[i, :] = x
        aux_variable[i-1,0]=0 #accept!
        i +=1
    return(np.hstack((samples,aux_variable)))



Data_results = Parallel(n_jobs=-1,max_nbytes=None)(delayed(torus_sampler)(n,R,r,θ,ϕ,σ,None)for σ in sigma_range)
Data_results=np.vstack(Data_results)
df= pd.DataFrame(Data_results)
df.rename(columns={0: "X", 1: "Y", 2: "Z",3: "status", 4: "scale"}, errors="raise",inplace=True)
df.to_csv('Range_Tor.csv',index=False)

