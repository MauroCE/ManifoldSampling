import math
import numpy as np
from numpy import zeros, eye, ones, log, exp, sqrt, diag, pi
from numpy.random import default_rng, randn
from numpy.linalg import norm
from scipy.stats import multivariate_normal as MVN
from scipy.optimize import fsolve
from warnings import catch_warnings, filterwarnings

from Manifolds.Manifold import Manifold


class LVManifold(Manifold):
    def __init__(self, Ns=50, step_size=1.0, σr=1.0, σf=1.0, r0=100, f0=100, z_true=(0.4, 0.005, 0.05, 0.001), seed=1111, seeds=[2222, 3333, 4444, 5555], n_chains=4):
        """Class defining the data manifold for the Lotka-Volterra ABC problem. The simulator is defined by an
        Euler-Marayama discretization of the LV SDE.

        Args:
            Ns (int, optional): Number of discretization time steps in the forward simulator. Defaults to 50.
            step_size (float, optional): Step size used in the discretization within the forward simulator. Defaults to 1.0.
            r0 (int, optional): Number of preys at time t=0. Defaults to 100.
            f0 (int, optional): Number of predators at time t=0. Defaults to 100.
            z_true (tuple, optional): True parameter values used to generate the data. Defaults to (0.4, 0.005, 0.05, 0.001).
            seed (int, optional): Random seed used to generate the data. Defaults to 1111.
            seeds (list, optional): List of seeds. Each seed used to find initial point for each chain. Defaults to [2222, 3333, 4444, 5555].
            n_chains (int, optional): Number of chains used to compute ESS using ArViz. Defaults to 4.
        """
        assert len(seeds) == n_chains, "Number of seeds must equal number of chains."
        self.Ns = Ns              # Number of steps used in integrating LV SDE
        self.m = 2*self.Ns        # Number of constraints = dimensionality of data
        self.d = 4                # Dimensionality of parameter
        self.n = self.d + self.m  # Dimensionality of ambient space
        self.δ = step_size        # Step size for discretization (not for sampling!)
        self.σr = σr              # Scale for noise in prey step
        self.σf = σf              # Scale for noise in predator step
        self.r0 = r0              # Initial prey population
        self.f0 = f0              # Initial predator population
        self.z_true = np.array(z_true)  # True parameter
        self.q_dist = MVN(zeros(self.n), eye(self.n))   # proposal for THUG
        self.seeds = seeds
        self.n_chains = n_chains 
        
        # generate data
        self.data_seed = seed
        self.rng = default_rng(self.data_seed)
        self.u1_true = self.z_to_u1(self.z_true)
        self.u2_true = self.rng.normal(loc=0.0, scale=1.0, size=2*self.Ns)
        self.u_true = np.concatenate((self.u1_true, self.u2_true))
        self.ystar  = self.u_to_x(self.u_true)
        
    def z_to_u1(self, z):
        """Transforms a parameter z into u1 (standard normal variables)."""
        assert len(z) == 4, "z should have length 4, but found {}".format(len(z))
        m_param = -2*ones(4)
        s_param = ones(4)
        return (log(z) - m_param) / s_param
    
    def u1_to_z(self, u1):
        """Given u1, it maps it to z."""
        assert len(u1) == 4, "u1 should have length 4, but found {}".format(len(u1))
        m_param = -2*ones(4)
        s_param = ones(4)
        return exp(s_param*u1 + m_param)
    
    def g(self, u):
        """Takes [u1, u2] and returns [z, u2]"""
        assert len(u) == self.n, "u should have length {}, but found {}".format(self.n, len(u))
        return np.concatenate((self.u1_to_z(u[:4]), u[4:]))
    
    def u_to_x(self, u):
        """Maps u=[u1, u2] to z."""
        assert len(u) == self.n, "u should have length {}, but found {}.".format(self.n, len(u))
        u1, u2 = u[:4], u[4:]
        u2_r   = u2[::2]
        u2_f   = u2[1::2]
        z1, z2, z3, z4 = self.u1_to_z(u1)
        r = np.full(self.Ns + 1, fill_value=np.nan)
        f = np.full(self.Ns + 1, fill_value=np.nan)
        r[0] = self.r0
        f[0] = self.f0
        for s in range(1, self.Ns+1):
            r[s] = r[s-1] + self.δ*(z1*r[s-1] - z2*r[s-1]*f[s-1]) + sqrt(self.δ)*self.σr*u2_r[s-1]
            f[s] = f[s-1] + self.δ*(z4*r[s-1]*f[s-1] - z3*f[s-1]) + sqrt(self.δ)*self.σf*u2_f[s-1]
        return np.ravel([r[1:], f[1:]], 'F')
    
    def zu2_to_x(self, zu2):
        """Same as u_to_x but this takes as input [z, u2]."""
        assert len(zu2) == self.n, "zu2 should have length {}, but found {}".format(self.n, len(zu2))
        z1, z2, z3, z4 = zu2[:4]
        u2 = zu2[4:]
        u2_r = u2[::2]
        u2_f = u2[1::2]
        r = np.full(self.Ns + 1, fill_value=np.nan)
        f = np.full(self.Ns + 1, fill_value=np.nan)
        r[0] = self.r0
        f[0] = self.f0
        for s in range(1, self.Ns+1):
            r[s] = r[s-1] + self.δ*(z1*r[s-1] - z2*r[s-1]*f[s-1]) + sqrt(self.δ)*self.σr*u2_r[s-1]
            f[s] = f[s-1] + self.δ*(z4*r[s-1]*f[s-1] - z3*f[s-1]) + sqrt(self.δ)*self.σf*u2_f[s-1]
        return np.ravel([r[1:], f[1:]], 'F')
    
    def Jg(self, ξ):
        """Jacobian of the function g:[u_1, u_2] --> [z, u_2]."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        m_param = -2*ones(4)
        s_param = ones(4)
        return diag(np.concatenate((s_param*exp(s_param*ξ[:4] + m_param), ones(2*self.Ns))))
    
    def oneat(self, ix, length=None):
        """Generates a vector of zeros of length `length` with a one at index ix."""
        assert type(ix) == int, "index for oneat() should be integer but found {}".format(type(ix))
        if length is None:
            length = self.n
        output = zeros(length)
        output[ix] = 1
        return output
    
    def Jf(self, ξ):
        """Jacobian of the function f:[z, u_2] --> x.
        Assume r and f contains r0 and f0 at the start."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        J = zeros((self.m, self.n))
        δ  = self.δ
        r0 = self.r0
        f0 = self.f0 
        σr = self.σr
        σf = self.σf
        # Sete first two rows: dr1_dξ and df1_dξ
        J[0, :] = np.concatenate(([δ*r0, -δ*r0*f0, 0, 0], sqrt(δ)*σr*self.oneat(0, length=self.m)))
        J[1, :] = np.concatenate(([0, 0, -δ*f0, δ*r0*f0], sqrt(δ)*σf*self.oneat(1, length=self.m)))
        # Evaluate function at the ξ to find r and f at this ξ.
        x = self.zu2_to_x(ξ)
        r = np.concatenate(([r0], x[::2]))
        f = np.concatenate(([f0], x[1::2]))
        # Grab the parameters
        z1, z2, z3, z4 = ξ[:4]
        # Loops through the time steps and compute the Markovian rows
        for s in range(1, self.Ns):
            J[2*s, :]     = J[2*s-2, :] + δ*(self.oneat(0)*r[s] + z1*J[2*s-2, :] -(self.oneat(1)*r[s]*f[s] + z2*J[2*s-2, :]*f[s] + z2*r[s]*J[2*s-1, :])) + sqrt(δ)*σr*self.oneat(2*s+4)
            J[2*s + 1, :] = J[2*s-1, :] + δ*(self.oneat(3)*r[s]*f[s] + z4*J[2*s-2, :]*f[s] + z4*r[s]*J[2*s-1, :] - self.oneat(2)*f[s] - z3*J[2*s-1, :]) + sqrt(δ)*σf*self.oneat(2*s+5)
        return J

    def q(self, ξ):
        """Constraint function taking u=[u1, u2] and comparing against true data."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        return self.u_to_x(ξ) - self.ystar
    
    def J(self, ξ):
        """Jacobian. Here u=[u1, u2]."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        return self.Jf(self.g(ξ)).dot(self.Jg(ξ))
    
    def Q(self, ξ):
        """Transpose of Jacobian."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        return self.J(ξ).T
    
    def logη(self, ξ):
        """Density on Manifold wrt Hausdorff measure."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        try:
            J = self.J(ξ)
            logprior = -ξ@ξ/2
            correction_term  = - math.prod(np.linalg.slogdet(J@J.T))/2 
            return  logprior + correction_term
        except ValueError as e:
            return -np.inf
        
    def find_point_on_manifold(self, maxiter=2000, tol=1e-14, random_u2_guess=False):
        """Finds a point on the Manifold with input u=[u1, u2]."""
        u2_guess = randn(self.m) if random_u2_guess else zeros(self.m)
        i = 0
        with catch_warnings():
            filterwarnings('error')
            while i <= maxiter:
                i += 1
                try: 
                    u1_init  = randn(self.d)*0.1 - 4
                    function = lambda u2: self.q(np.concatenate((u1_init, u2)))
                    fprime   = lambda u2: self.J(np.concatenate((u1_init, u2)))[:, self.d:]
                    u2_found = fsolve(function, u2_guess, xtol=tol, fprime=fprime)
                    u_found = np.concatenate((u1_init, u2_found))
                    return u_found
                except RuntimeWarning:
                    continue
        raise ValueError("Couldn't find a point, try again.")
        
    def find_point_on_manifold_given_u1true(self, maxiter=2000, tol=1e-14, random_u2_guess=False):
        """Finds a point on the Manifold starting from u1_true."""
        i = 0
        with catch_warnings():
            filterwarnings('error')
            while i <= maxiter:
                i += 1
                try:
                    u2_guess = randn(self.m) if random_u2_guess else zeros(self.m)
                    function = lambda u2: self.q(np.concatenate((self.u1_true, u2)))
                    u2_found = fsolve(function, u2_guess, xtol=tol)
                    u_found = np.concatenate((self.u1_true, u2_found))
                    return u_found
                except RuntimeWarning:
                    continue
        raise ValueError("Couldn't find a point, try again.")

    def find_init_points_for_each_chain(self, u1_true=True, random_u2_guess=False, tol=1e-14, maxiter=5000):
        """Finds `n_chains` initial points on the manifold.

        Args:
            u1_true (boool, optional): Whether to use u1 that generated the data or sample it at random.
            random_u2_guess (bool, optional): Whether to generate the initial u2 guess at random or as a zero vector. Defaults to False.
            tol (float, optional): tolerance for fsolve. Defaults to 1e-14.
            maxiter (int, optional): Maximum number of iterations for optimization procedure. Defaults to 5000.

        Returns:
            ndarray: array having dimension (n_chains, n), containing each point on a row.
        """
        u0s = zeros((self.n_chains, self.n))
        for i in range(self.n_chains):
            if u1_true:
                u0s[i, :] = self.find_point_on_manifold_given_u1true(maxiter=maxiter, tol=tol, random_u2_guess=random_u2_guess)
            else:
                u0s[i, :] = self.find_point_on_manifold(maxiter=maxiter, tol=tol, random_u2_guess=random_u2_guess)
        self.u0s = u0s 
        return self.u0s
            
    def transform_usamples_to_zsamples(self, samples):
        """Given samples of size (N, 4 + 2*Ns) it takes the first 4 columns and transforms them."""
        n_samples, input_dim = samples.shape
        assert input_dim == self.n, "Wrong dim. Expected {} , found {}".format(self.n, input_dim)
        return np.apply_along_axis(self.u1_to_z, 1, samples[:, :4])
    
    def log_normal_kernel(self, ξ, ϵ):
        """Log normal kernel density."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        u = norm(self.q(ξ))   ##### THIS IS NOT THE USUAL u
        return -u**2/(2*(ϵ**2)) -0.5*log(2*pi*(ϵ**2))

    def generate_logpi(self, ϵ):
        """Generates ABC posterior using a certain epsilon value. Uses a Gaussian kernel. """
        logηϵ = lambda ξ: self.log_normal_kernel(ξ, ϵ) - ξ@ξ/2
        return logηϵ
    
    def is_on_manifold(self, ξ, tol=1e-14):
        """Checks if a point is on the manifold."""
        return max(abs(self.q(ξ))) <= tol