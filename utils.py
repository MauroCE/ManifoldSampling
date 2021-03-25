from scipy.stats import multivariate_normal
import scipy
import numpy as np
import plotly.graph_objects as go
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt


def logf(xyz):
    """
    This function is usually used as the uniform target distribution on the manifold
    """
    return 0

def logp(xyz, sigma=0.5):
    """
    This function is used as proposal distribution. It is simply a 2D isotropic 
    normal distribution with scale sigma.
    """
    return multivariate_normal.logpdf(xyz, mean=np.zeros(2), cov=(sigma**2)*np.eye(2))

def quick_3d_scatter(samples):
    """
    Plots 3D samples using plotly.
    """
    fig = go.Figure(
    data=[
        go.Scatter3d(
            x=samples[:,0], 
            y=samples[:,1], 
            z=samples[:,2], 
            mode="markers",
            marker=dict(
                size=1.0,
                opacity=0.8
            ))
        ]
    )
    fig.show()

def quick_MVN_scatter(samples, target, xlims=[-2, 6], ylims=[-3, 5], figsize=(20, 8), lw=5):
    """
    Plots 2D samples and contours of MVN.
    """
    # Grid of points for contour plot
    x, y = np.mgrid[xlims[0]:xlims[1]:.01, ylims[0]:ylims[1]:.01]
    pos = np.dstack((x, y))

    fig, ax = plt.subplots(figsize=figsize)
    ax.contour(x, y, target.pdf(pos), linewidths=lw) 
    ax.scatter(*samples.T)
    plt.show()

def quick_MVN_marginals(samples, target, lims=(-4,4), figsize=(20,5), n=100, bins=50):
    """
    Plots marginals.
    """
    marginal_x = lambda x: scipy.stats.norm(loc=target.mean[0], scale=np.sqrt(target.cov[0, 0])).pdf(x)
    marginal_y = lambda y: scipy.stats.norm(loc=target.mean[1], scale=np.sqrt(target.cov[1, 1])).pdf(y)

    x = np.linspace(lims[0], lims[1], num=n)

    fig, ax = plt.subplots(ncols=2, figsize=figsize)
    # X marginal
    ax[0].plot(x, marginal_x(x))
    _ = ax[0].hist(samples[:, 0], density=True, bins=bins)
    # Y marginal
    ax[1].plot(x, marginal_y(x))
    _ = ax[1].hist(samples[:, 1], density=True, bins=bins)
    plt.show()




def normalize(x):
    """
    Normalizes a vector.
    """
    return x / np.sqrt(np.sum(x**2))


def logf_Jacobian(xy, Sigma):
    """
    1 / Jacobian of log pi
    """
    return np.log(1 / norm(inv(Sigma) @ xy)) # 1 / norm(inv(Sigma) @ xy)
