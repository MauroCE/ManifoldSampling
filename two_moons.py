import numpy as np
from math import sqrt, cos, sin


def tm_deterministic(xi):
    """Two Moons deterministic simulator"""
    # Extract theta1, theta2 and seeds u = (a,r)
    t0, t1, a, r = xi[0], xi[1], xi[2], xi[3]
    # Run simulator
    p = np.array([r * np.cos(a) + 0.25, r * np.sin(a)])
    return p + np.array([-np.abs(t0 + t1), (-t0 + t1)]) / sqrt(2)


def tm_jacobian(xi):
    """Jacobian of the simulator. Since f: R^4 -> R^2
    the Jacobian should be R^2 -> R^4. Notice Zappa's algorithm
    requires this to be the transpose of the Jacobian."""
    t1, t2, a, r = xi
    val = - (t1 + t2) / (abs(t1 + t2) * sqrt(2))
    return np.array([
        [val, val, -r * sin(a), cos(a)],
        [-1/sqrt(2), 1/sqrt(2), r * cos(a), sin(a)]
    ])


def tm_distance_gradient(xi, y_star, eps):
    """Grandient of the distance manifold. Since distance manifold constraint function
    is a function f: R^4 -> R Jacobian should be (1, 4). Most likely this
    gradient should be rescaled."""
    return (tm_jacobian(xi).T @ (tm_deterministic(xi) - y_star) / eps).reshape(1, 4)
