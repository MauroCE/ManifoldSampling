import numpy as np
from Manifolds.Manifold import Manifold
import matplotlib.pyplot as plt

class gk(Manifold):
    def __init__(self, ystar):
        """
        Class for the g and k model collecting functions and information

        ystar=data observed
        """
        super().__init__(m=1, d=5)
        self.ystar = ystar

    def Q(self, T):
        """Q"""
        dfda=1
        dfdb=T[5]*(1+T[5]**2)**(T[4])*(1+np.tanh(T[5]*T[3]/2))
        dfdc=T[1]*T[5]*(1+T[5]**2)**(T[4])*np.tanh(T[5]*T[3]/2)
        dfdg=T[1]*T[5]*(1+T[5]**2)**(T[4])*(1-(np.tanh(T[5]*T[3]/2)**2))*T[2]*T[5]/2
        dfdk=T[1]*T[5]*(1+np.tanh(T[5]*T[3]/2))*T[4]*(1+T[5]**2)**(T[4])*np.log(1+T[5]**2)
        dfdu=T[1]*(1+T[5]**2)**(T[4])*(1+np.tanh(T[5]*T[3]/2))+T[1]*T[5]*T[4]*2*T[5]*(1+np.tanh(T[5]*T[3]/2))+T[1]*(1+T[5]**2)**(T[4])**(1-(np.tanh(T[5]*T[3]/2)**2))*T[2]*T[3]/2
        return np.array([dfda,dfdb,dfdc,dfdg,dfdk,dfdu]).reshape(-1, self.m)
    
    def q(self, T):
        """Constraint function for the g-and-k model with data observed y_star"""
        return T[0]+T[1]*T[5]*(1+T[5]**2)**(T[4])*(1+np.tanh(T[5]*T[3]/2))- self.ystar

    