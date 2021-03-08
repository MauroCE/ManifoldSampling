from scipy.stats import multivariate_normal
import numpy as np
import plotly.graph_objects as go


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

def normalize(x):
    """
    Normalizes a vector.
    """
    return x / np.sqrt(np.sum(x**2))