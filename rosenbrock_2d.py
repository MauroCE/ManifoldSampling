from numpy import exp, ndarray, array


class RosenbrockDensity2D:

    def __init__(self, c=20):
        """C is the constant we use to divide."""
        self.c = c 
    
    def f(self, x):
        """Evaluates Rosenbrock density on an input np.array([x1, x2])."""
        if type(x) != ndarray:
            raise TypeError("Input must be a numpy array.")
        else:
            if x.shape == (2,):
                # Then we have a single input
                return exp(-(100*(x[1] - x[0]**2)**2 + (1-x[0])**2) / self.c)
            elif len(x.shape) == 2 and x.shape[1] == 2 and x.shape[0] >= 1:
                return exp(-(100*(x[:, 1] - x[:, 0]**2)**2 + (1-x[:, 0])**2) / self.c)
            else:
                raise ValueError("Input must have shape either (2, ) or (n, 2) but found {}".format(x.shape))

    def logf(self, x):
        """Log Rosenbrock density."""
        if type(x) != ndarray:
            raise TypeError("Input must be a numpy array.")
        else:
            if x.shape == (2,):
                # Then we have a single input
                return -(100*(x[1] - x[0]**2)**2 + (1-x[0])**2) / self.c
            elif len(x.shape) == 2 and x.shape[1] == 2 and x.shape[0] >= 1:
                return -(100*(x[:, 1] - x[:, 0]**2)**2 + (1-x[:, 0])**2) / self.c
            else:
                raise ValueError("Input must have shape either (2, ) or (n, 2) but found {}".format(x.shape))

    def grad_logf(self, x):
        """Gradient of Rosenbrock function."""
        x1, x2 = x
        gradx1 = 400*x1*(x2 - x1**2)/self.c + 2*(1 - x1)/self.c
        gradx2 = -200*(x2 - x1**2)/self.c
        return array([gradx1, gradx2])
