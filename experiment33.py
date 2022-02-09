# Experiment 33 (UNFINISHED DUE TO GRADIENT): HH vs TH on Lotka-Volterra model.
import numpy as np
from numpy import sqrt, zeros, exp
from numpy.random import exponential, rand
from numpy.linalg import norm

from scipy.stats import norm as ndist


def findfirst(bool_array):
    """Finds index of first True element in bool_array."""
    if bool_array.any():
        return np.arange(len(bool_array))[bool_array][0]
    else:
        print("No element satisfies condition.")
        return None

def findlast(bool_array):
    "Finds index of last True element in bool_array."
    if bool_array.any():
        return np.arange(len(bool_array))[bool_array][-1]
    else:
        print("No element satisfies condition.")
        return None


def LVsimulator(theta, tmax, dt=1.0, x0=100, y0=100):
    """Simulates Lotka-Volterra."""
    xvec = [x0]
    yvec = [y0]
    tvec = [0.0]            # Vector of times
    nlat = 0                # Number of latents
    while True:
        # Compute rate to next event, sample waiting time and add next event time to storage
        rate_vec = np.array([theta[0]*xvec[-1]*yvec[-1], theta[1]*xvec[-1], theta[2]*yvec[-1], theta[3]*xvec[-1]*yvec[-1]])
        waiting_time = exponential(1 / rate_vec.sum())
        nlat += 1
        tvec.append(tvec[-1] + waiting_time)

        # Stop simulating if exceeded terminal time
        if tvec[-1] > tmax: break

        event_ix = findfirst(rand() <= (np.cumsum(rate_vec) / np.sum(rate_vec)))

        nlat += 1
        if event_ix == 1:
            xvec.append(xvec[-1] + 1) # X population increases
            yvec.append(yvec[-1])     
        elif event_ix == 2:
            xvec.append(xvec[-1] - 1) # X population decreases
            yvec.append(yvec[-1])     
        elif event_ix == 3:
            xvec.append(xvec[-1])
            yvec.append(yvec[-1] + 1) # Y population increases
        else:
            xvec.append(xvec[-1])
            yvec.append(yvec[-1] - 1) # Y population decreases
        
        obst = np.arange(dt, tmax + dt, dt)  # Times at which we observe the process
        obsx = zeros(len(obst))
        obsy = zeros(len(obst))
        for i in range(len(obst)):
            ix = findlast(tvec < obst[i])
            obsx[i] = xvec[ix]
            obsy[i] = yvec[ix]
        return {
            'x': xvec[1:], 
            'y': yvec[1:], 
            't': tvec[1:-1],
            'obsx': obsx, 
            'obsy': obsy, 
            'obst': obst, 
            'nlat': nlat
        }


def f(xi, x0=100, y0=100, dt=1.0, sx=1.0, sy=1.0):
    """Deterministic function for LV simulator. Here xi = (logtheta, u)
    because we assign a logNormal prior on the parameters, so it is easier to
    work with logtheta."""
    logtheta, u = xi[:4], xi[4:]
    t1, t2, t3, t4 = exp(logtheta)
    N = len(u) // 2
    xvec = zeros(N+1)
    yvec = zeros(N+1)
    xvec[0] = x0
    yvec[0] = y0
    for i in range(N):
        xvec[i+1] = max(0, xvec[i] + dt * (t1*xvec[i] - t2*xvec[i]*yvec[i]) + sqrt(dt) * sx * u[2*i])
        yvec[i+1] = max(0, yvec[i] + dt * (t4*xvec[i]*yvec[i] - t3*yvec[i]) + sqrt(dt) * sy * u[2*i+1])
    return np.concatenate((xvec[1:], yvec[1:]))


def fnorm(xi, ystar):
    return norm(f(xi) - ystar)


def logprior(xi):
    """logtheta ~ N(-2, 1.0) for each theta. Remember xi contains
    logtheta and not theta. We assign a normal prior to the latents."""
    logtheta, u = xi[:4], xi[4:]
    return ndist(loc=-2, scale=1.0).logpdf(logtheta).sum() + ndist.logpdf(xi[4:]).sum()



if __name__ == "__main__":
    n_samples = 1000
    theta = np.array([0.01, 0.5, 1.0, 0.01])
    NlatVec = zeros(n_samples)
    for i in range(n_samples):
        data = LVsimulator(theta, tmax=50, x0=50, y0=100)
        NlatVec[i] = data['nlat']
        print(i)
    
