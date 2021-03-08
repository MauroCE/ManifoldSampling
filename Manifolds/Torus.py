import numpy as np
from Manifolds.Manifold import Manifold
import matplotlib.pyplot as plt

class Torus(Manifold):
    def __init__(self, mu, R, r):
        """
        Class for a torus. It collects functions and information that relates to a torus.

        mu : Numpy Array
             Center of the torus. Must be a 1D array.
        R : Float
            Toroidal radius.
        r : Float
            Poloidal radius
        """
        super().__init__(m=1, d=2)
        self.mu = mu
        self.R = R
        self.r = r

    def to_cartesian(self, theta_phi):
        """
        Takes in a point as a 1D array containing theta and phi and returns its version in 
        Cartesian coordinates.

        theta_phi : Numpy Array
                    1D Array of size (2, ) where theta_phi[0] is theta and theta_phi[1] is phi.
        returns : Its version in Cartesian coordinates.
        """
        theta, phi = theta_phi[0], theta_phi[1]
        x = self.mu[0] + (self.R + self.r * np.cos(theta)) * np.cos(phi)
        y = self.mu[1] + (self.R + self.r * np.cos(theta)) * np.sin(phi)
        z = self.mu[2] + self.r * np.sin(theta)
        return np.array([x, y, z])

    def q(self, xyz):
        """
        Constraint function defining the Torus.

        xyz : Numpy Array
              1D Numpy Array of dimension (3, ) in Cartesian coordinates and returns a 1D array with one element
              corresponding to the value of the constraint at that point. If q(xyz) = 0 then we are on the Torus.
        """
        xc, yc, zc = xyz - self.mu   # center
        return (np.sqrt(xc**2 + yc**2) - self.R)**2 + zc**2 - self.r**2

    def Q(self, xyz):
        """
        Computes the Q matrix needed for Miranda's algorithm. This is the transpose of the Jacobian. The matrix Q
        will have dimension (3, 1). Importantly, notice how the output is reshaped to (-1, 1) before being returned.

        xyz : Numpy Array
            1D array of dimensions (3, ) containing a point in Cartesian space at which we want to compute the matrix Q.

        return : (3, 1) matrix Q
        """
        xc, yc, zc = xyz - self.mu   # Center
        return np.array([
            2 * xc * (np.sqrt(xc**2 + yc**2) - self.R) / np.sqrt(xc**2 + yc**2),
            2 * yc * (np.sqrt(xc**2 + yc**2) - self.R) / np.sqrt(xc**2 + yc**2),
            2 * zc
        ]).reshape(-1, self.m)

    def find_phi(self, y, x, already_centered=True):
        """
        Essentialy this is arctan2 bu correct for the torus. 

        y : Numpy Array
            y coordinate for the point to which we want to find phi. Since this is parallelizable, it can be 
            samples[:, 1] where `samples` is the output of Miranda's algorithm.

        x : Numpy Array
            Same but for x coordinate. (and, correspondingly samples[:,0]).

        already_centered : Whether y and x are already centered or not.
        """
        if not already_centered:
            y = y - self.mu[1]
            x = x - self.mu[0]
        return (2*np.pi + np.arctan2(y, x)) % (2*np.pi)


    def find_theta(self, xyz, already_centered=True):
        """
        If you are on the outside, you have to shift [-pi/2, pi/2] to [0, pi/2] U [3pi/2, 2pi].
        If you are on the inside, you have to shift [-pi/2, pi/2] to [pi/2, pi] U [pi, 3pi/2].
        XYZ MUST BE CENTERED.
        """
        if not already_centered:
            xyz = xyz - self.mu
        x, y, z = xyz
        if (x**2 + y**2) >= self.R**2:
            return (np.arcsin(z/self.r) + 2*np.pi) % (2*np.pi)
        else:
            return np.pi - np.arcsin(z/self.r)

    def plot_marginals(self, samples, bins, thinning):
        """
        Plots the phi and theta marginals.
        """
        # Center the samples
        samples = samples - self.mu

        # Angles for the x axis
        x = np.linspace(0, 2*np.pi + (2*np.pi)/100, 100)

        # Phi and Theta values from samples
        phis = self.find_phi(samples[:, 1], samples[:, 0])
        thetas = np.apply_along_axis(self.find_theta, 1, samples)

        fig, ax = plt.subplots(ncols=2, figsize=(10, 6))
        # Plot for phi
        _ = ax[0].hist(phis[::thinning], bins=bins, density=True)    # Histogram
        ax[0].plot(x, np.repeat(1/(2*np.pi), 100))               # True Marginal
        # Plot for theta
        _ = ax[1].hist(thetas[::thinning], bins=bins, density=True)
        ax[1].plot(x, (1 + (self.r/self.R)*np.cos(x))/(2*np.pi))
        # Prettify
        ax[0].set_xlabel(r"$\phi$", fontsize=18)
        ax[1].set_xlabel(r"$\theta$", fontsize=18)
        plt.show()


