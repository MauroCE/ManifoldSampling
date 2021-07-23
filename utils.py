from scipy.stats import multivariate_normal
import scipy
import numpy as np
import plotly.graph_objects as go
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import expon
from oct2py import octave


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

def logpexp_scale(xyz, scale=1.0):
    """Exponential proposal log density"""
    return expon.logpdf(xyz, scale=scale)

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


def quick_3d_scatters(samples, labels, size=1.0, opacity=0.8):
    """
    Multiple 3D scatter plots in the same figure.
    """
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=sample[:,0], 
                y=sample[:,1], 
                z=sample[:,2], 
                mode="markers",
                name='{}'.format(label),
                marker=dict(
                    size=size,
                    opacity=opacity
                )) for (sample, label) in zip(samples, labels)
        ]
    )
    fig.show()


def quick_MVN_scatter(samples, target, xlims=[-2, 6], ylims=[-3, 5], figsize=(20, 8), lw=5, levels=None, alpha=1.0, zorder=1, colors='gray', step=0.01, return_axes=False):
    """
    Plots 2D samples and contours of MVN.
    """
    # Grid of points for contour plot
    x, y = np.mgrid[xlims[0]:xlims[1]:step, ylims[0]:ylims[1]:step]
    pos = np.dstack((x, y))

    fig, ax = plt.subplots(figsize=figsize)
    if levels is None:
        ax.contour(x, y, target.pdf(pos), linewidths=lw) 
    else:
        ax.contour(x, y, target.pdf(pos), linewidths=lw, levels=levels, alpha=alpha, zorder=1, colors=colors) 
    ax.scatter(*samples.T)
    if not return_axes:
        plt.show()
    else:
        return fig, ax


def MVN_scatters(samples_list, target, xlims=[-2, 6], ylims=[-3, 5], figsize=(20, 8), lw=5, levels=None, alpha=1.0, zorder=1, colors='gray', step=0.01, return_axes=False, labels=None):
    """
    Plots 2D samples and contours of MVN.
    """
    # Grid of points for contour plot
    x, y = np.mgrid[xlims[0]:xlims[1]:step, ylims[0]:ylims[1]:step]
    pos = np.dstack((x, y))

    fig, ax = plt.subplots(figsize=figsize)
    if levels is None:
        ax.contour(x, y, target.pdf(pos), linewidths=lw) 
    else:
        ax.contour(x, y, target.pdf(pos), linewidths=lw, levels=levels, alpha=alpha, zorder=1, colors=colors) 
    for ix, samples in enumerate(samples_list):
        if labels is None:
            ax.scatter(*samples.T)
        else:
            ax.scatter(*samples.T, label=labels[ix])
            ax.legend()
    if not return_axes:
        plt.show()
    else:
        return fig, ax





def quick_MVN_marginals(samples, target, xlims=(-4,4), ylims=(-4,4), figsize=(20,5), n=100, bins=50):
    """
    Plots marginals.
    """
    marginal_x = lambda x: scipy.stats.norm(loc=target.mean[0], scale=np.sqrt(target.cov[0, 0])).pdf(x)
    marginal_y = lambda y: scipy.stats.norm(loc=target.mean[1], scale=np.sqrt(target.cov[1, 1])).pdf(y)

    x = np.linspace(xlims[0], xlims[1], num=n)
    y = np.linspace(ylims[0], ylims[1], num=n)

    fig, ax = plt.subplots(ncols=2, figsize=figsize)
    # X marginal
    ax[0].plot(x, marginal_x(x))
    _ = ax[0].hist(samples[:, 0], density=True, bins=bins)
    #ax[0].set_aspect("equal")
    # Y marginal
    ax[1].plot(y, marginal_y(y))
    _ = ax[1].hist(samples[:, 1], density=True, bins=bins)
    #ax[1].set_aspect("equal")
    plt.show()



def quick_MGM_marginals(samples, target, xlims=(-4,4), ylims=(-4,4), figsize=(20,5), n=100, bins=50):
    """
    Plots marginals.
    """
    marginal_x = lambda x: scipy.stats.norm(loc=target.mean[0], scale=np.sqrt(target.cov[0, 0])).pdf(x)
    marginal_y = lambda y: scipy.stats.norm(loc=target.mean[1], scale=np.sqrt(target.cov[1, 1])).pdf(y)

    x = np.linspace(xlims[0], xlims[1], num=n)
    y = np.linspace(ylims[0], ylims[1], num=n)

    fig, ax = plt.subplots(ncols=2, figsize=figsize)
    # X marginal
    ax[0].plot(x, marginal_x(x))
    _ = ax[0].hist(samples[:, 0], density=True, bins=bins)
    #ax[0].set_aspect("equal")
    # Y marginal
    ax[1].plot(y, marginal_y(y))
    _ = ax[1].hist(samples[:, 1], density=True, bins=bins)
    #ax[1].set_aspect("equal")
    plt.show()




def quick_MVN_marginals_kde(samples, target, lims=(-4, 4), figsize=(20, 5), n=100, bins=50):
    """
    Plots KDE.
    """
    # KDE estimators
    xkde = gaussian_kde(samples[:, 0])
    ykde = gaussian_kde(samples[:, 1])
    # True Marginals
    marginal_x = lambda x: scipy.stats.norm(loc=target.mean[0], scale=np.sqrt(target.cov[0, 0])).pdf(x)
    marginal_y = lambda y: scipy.stats.norm(loc=target.mean[1], scale=np.sqrt(target.cov[1, 1])).pdf(y)
    # Data for plot
    x = np.linspace(lims[0], lims[1], num=n)
    # plot
    fig, ax = plt.subplots(ncols=2, figsize=figsize)
    ax[0].plot(x, marginal_x(x))
    ax[0].plot(x, xkde(x))
    ax[1].plot(x, marginal_y(x))
    ax[1].plot(x, ykde(x))
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


def prep_contour(xlims, ylims, step, func):
    x = np.arange(*xlims, step)
    y = np.arange(*ylims, step)
    x, y = np.meshgrid(x, y)
    xshape = x.shape
    xfl, yfl = x.flatten(), y.flatten()
    xys = np.vstack((xfl, yfl)).T
    return x, y, func(xys).reshape(xshape)


def update_scale_sa(ap, ap_star, k, l, exponent=(2/3)):
    """
    Updates the scale in adaptive zappa.

    ap : float
         Current acceptance probability

    ap_star : float
              Target acceptance probability.

    k : int 
        Iteration number. Notice that it must start from 1, not 0!

    l : float
        Current value of log scale

    exponent : float
               Exponent for the step size.

    Returns
    -------
    s : float
        Updated exponential scale value
    l : float
        Updated log scale value
    """
    step_size = 1 / k ** exponent
    l = l + step_size * (ap - ap_star)
    return np.exp(l), l


def angle_between(v1, v2):
    """
    Computes angle in radiant between two n-dimensional vectors.
    """
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def covariance(samples):
    """Computes covariance between samples."""
    X = samples - np.mean(samples, axis=0)
    return X.T @ X / (samples.shape[0] - 1)



ESS = lambda samples: octave.multiESS(samples, [], "sqroot")


def quick_MVN_marginals_kdes(sample_list, target, labels, lims=(-4, 4), figsize=(20, 5), n=100):
    """
    Plots KDE.
    """
    # True Marginals
    marginal_x = lambda x: scipy.stats.norm(loc=target.mean[0], scale=np.sqrt(target.cov[0, 0])).pdf(x)
    marginal_y = lambda y: scipy.stats.norm(loc=target.mean[1], scale=np.sqrt(target.cov[1, 1])).pdf(y)
    # Data for plot
    x = np.linspace(lims[0], lims[1], num=n)
    # plot
    fig, ax = plt.subplots(ncols=2, figsize=figsize)
    ax[0].plot(x, marginal_x(x))
    ax[1].plot(x, marginal_y(x))
    for ix, samples in enumerate(sample_list):
        # KDE estimators
        xkde = gaussian_kde(samples[:, 0])
        ykde = gaussian_kde(samples[:, 1])
        # Add to plot
        ax[0].plot(x, xkde(x), label=labels[ix])
        ax[1].plot(x, ykde(x), label=labels[ix])
    plt.legend()
    plt.show()