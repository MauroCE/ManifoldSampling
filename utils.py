from scipy.stats import multivariate_normal
import scipy
import numpy as np
import plotly.graph_objects as go
from numpy.linalg import norm, inv, solve
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
from scipy.stats import gaussian_kde
from scipy.stats import expon
from oct2py import octave
import tensorflow_probability as tfp
from arviz import ess as ess_arviz
from arviz import convert_to_dataset


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


def quick_MVN_scatter(samples, target, xlims=[-2, 6], ylims=[-3, 5], figsize=(20, 8), lw=5, levels=None, alpha=1.0, zorder=1, colors='gray', step=0.01, return_axes=False, aspect=False):
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
    if aspect:
        ax.set_aspect("equal")
    if not return_axes:
        plt.show()
    else:
        return fig, ax


def MVN_scatters(samples_list, target, xlims=[-2, 6], ylims=[-3, 5], figsize=(20, 8), lw=5, levels=None, alpha=1.0, zorder=1, colors='gray', step=0.01, return_axes=False, labels=None, axis=None):
    """
    Plots 2D samples and contours of MVN.
    """
    # Grid of points for contour plot
    x, y = np.mgrid[xlims[0]:xlims[1]:step, ylims[0]:ylims[1]:step]
    pos = np.dstack((x, y))

    if axis is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = axis

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
        if axis is None:
            return fig, ax
        else:
            return ax



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


def logf_Jacobian(xy, Sigma, mu):
    """
    1 / Jacobian of log pi
    """
    return - np.log(norm(solve(Sigma, xy - mu)))


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


def ESS(samples):
    """Computes multiESS using MATLAB function. Sometimes if the
    samples array has 1 unique sample the output will be complex. In that case we return 0.0 instead.
    Shall I output 1.0 or 0.0? Maybe 0.0 makes more sense?"""
    ESSval = octave.multiESS(samples, [], "sqroot")
    return ESSval #if type(ESSval) == float else 0.0

ESS_univariate = lambda samples: tfp.mcmc.effective_sample_size(samples).numpy()

def n_unique(samples):
    return np.unique(samples, axis=0).shape[0]

def ESS_times_proportion_unique(samples, axis):
    return ESS_univariate(samples[:, axis]) * (n_unique(samples) / len(samples))


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
        ax[0].legend()
        ax[1].plot(x, ykde(x), label=labels[ix])
        ax[1].legend()
    plt.show()


def box_plot(ax, data, edge_color, fill_color, positions, labels=None, widths=0.2):
    bp = ax.boxplot(data, patch_artist=True, positions=positions, widths=widths)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)

    for patch in bp['fliers']:
        patch.set(markeredgecolor=edge_color)
    return bp


def num_grad_hug_hop(N, B):
    return (B * N / 2) + N

def num_grad_thug_hop(N, B):
    return ((B + 2) * N / 2) + N


def test_circle(x0, n, q, δ, α, grad_logπ, logπ):
    """
    This function checks if HUG and THUG always end up on the circle manifold after 1 iteration.
    For n times it starts at x0, perform one HUG or THUG step (with B=1) and records the end position.
    At the end the function checks if all these end positions are on the circle.
    """
    hug_moves = []
    thug_moves = []
    # Perform n steps of hug with B=1
    for _ in range(n):
        v0 = q.rvs()
        x = x0 + δ*v0/2
        g = grad_logπ(x)
        ĝ = g / norm(g)
        v = v0 - 2*(v0 @ ĝ) * ĝ
        hug_moves.append(x + δ*v/2)
    # Perform n steps of thug with B=1
    for _ in range(n):
        v0s = q.rvs()
        g = grad_logπ(x0)
        g = g / norm(g)
        v0 = v0s - α*g*(g @ v0s)
        x = x0 + δ*v0/2
        g = grad_logπ(x)
        ĝ = g / norm(g)
        v = v0 - 2*(v0 @ ĝ) * ĝ
        thug_moves.append(x + δ*v/2)
    return max(abs(logπ(hug_moves) - logπ(x0))), max(abs(logπ(thug_moves) - logπ(x0)))


def ar_and_var_change_for_hug_thug(x0, T, B, N, αs, q, logπ, grad_logπ, Hug, HugTangential):
    """
    For different alphas, this function runs Hug and Thug and computes the decrease in Acceptance
    Rate and the improvement in variance brought by Thug.
    """
    # For different alphas I see what is the acceptance rate decrease and the variance improvement
    ar_decreases = []
    var_improv = []
    s_hugs = []
    s_thugs = []
    for α in αs:
        samples_hug, accept_hug = Hug(x0, T, B, N, q, logπ, grad_logπ)
        samples_thug, accept_thug = HugTangential(x0, T, B, N, α, q, logπ, grad_logπ)
        var_hug = np.var(logπ(samples_hug) - logπ(x0))
        var_thug = np.var(logπ(samples_thug) - logπ(x0))
        ar_decreases.append((np.sum(accept_hug) - np.sum(accept_thug)) * 100 / np.sum(accept_hug))
        var_improv.append(var_hug / var_thug)
        s_hugs.append(samples_hug)
        s_thugs.append(samples_thug)
    return ar_decreases, var_improv, s_hugs, s_thugs


invert_sign = lambda x: 1*x - 2*x

rangeof = lambda x: (np.min(x), np.max(x))


def line_perp_v_through_point(v, point, xvalues):
    """Returns yvalues corresponding to xvalues for a line perpendicular to v and passing through point. Example:
    plt.plot(xvalues, *line_perp_v_through_point(v, x, xvalues))
    """
    m = - v[0] / v[1]
    q = (v @ point) / v[1]
    return m*xvalues + q


def line_between(point1, point2):
    """Returns array that can be used to plot line between point1 and point2. Example:
    ```
    plt.plot(*line_between(point1, point2), color='k', lw=2)
    ```
    """
    return np.vstack((point1, point2)).T


def compute_arviz_miness_runtime(chains, times):
    """Computes minESS/runtime. Expects chains=[samples, samples, ...] and times = [time, time, ...]."""
    assert np.all([chain.shape == chains[0].shape for chain in chains]), "Chains must have same dimensions."
    n_samples = len(chains[0])
    stacked = np.vstack([chain.reshape(1, n_samples, -1) for chain in chains])
    dataset = convert_to_dataset(stacked)
    return min(np.array(ess_arviz(dataset).to_array()).flatten()) / np.mean(times)


def generate_powers_of_ten(max_exponent, min_exponent):
    """E.g. generate_powers_of_ten(2, -1) will return 100, 10, 0, 0.1."""
    number_of_powers = max_exponent + abs(min_exponent) + 1
    return np.logspace(start=max_exponent, stop=min_exponent, num=number_of_powers, endpoint=True)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def adagrad(function, gradient, initial_point, learning_rate=0.05, tolerance=1e-8, max_iter=10000):
    """Performs AdaGrad, an adaptive gradient descent optimization strategy to
    minimize the function `function` with gradient `gradient`, using a learning
    rate `learning_rate`, a tolerance `tolerance` and a maximum number of iterations 
    `max_iter`, starting from `initial_point`."""
    x = np.array(initial_point)
    grad_squared_sum = np.zeros_like(x)

    for _ in range(max_iter):
        grad = gradient(x)
        grad_squared_sum += grad**2
        x -= (learning_rate / (np.sqrt(grad_squared_sum) + tolerance)) * grad

        if np.linalg.norm(grad) < tolerance:
            break

    return x
