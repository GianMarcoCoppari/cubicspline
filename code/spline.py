"""
Defines the CubicSpline class.
"""

import numpy as np
from .lu import lu
from .tls import solver

class CubicSpline():
    """
    CubicSpline class.

    Parameters
    --------------
    X : np.array
        Containts x values of spline nodes.
    Y : np.array
        Contains y values of spline nodes.
    BC : np.array
        Contains left and right boundary conditions. 
    """

    def __init__(self, X: np.array, Y: np.array, BC: np.array):
        """
        CubicSpline constructor. Intanciate a CubicSpline object, computing all the coefficients.

        Parameters
        --------------
        X : np.array
            Containts x values of spline nodes.
        Y : np.array
            Contains y values of spline nodes.
        BC : np.array
            Contains left and right boundary conditions.

        Returns
        ---------------
        CubicSpline
            Returns an instance of the class.

        Raises
        ---------------
        AssertionError: if one of the following conditions are met
            - the number of nodes is less than two.
            - size of X and Y numpy arrays do not match.
            - the x values of the nodes are not unique.
            - more than two boundary conditions are given.
            - x values for nodes are not ordered.
        """
        if len(X) < 2:
            raise AssertionError
        if len(X) != len(Y):
            raise AssertionError
        if len(X) != len(np.unique(X)):
            raise AssertionError
        if not np.all(np.diff(X) > 0):
            raise AssertionError
        if len(BC) != 2:
            raise AssertionError

        self.nodes = X
        self.size = len(X) - 1
        self.params = [
            np.zeros(self.size),
            np.zeros(self.size),
            np.zeros(self.size),
            np.zeros(self.size)]
        

        dx = np.array([X[i + 1] - X[i] for i in range(self.size)])
        dy = np.array([Y[i + 1] - Y[i] for i in range(self.size)])
        

        if len(X) == 2:
            v, u, w = np.zeros(self.size), np.zeros(self.size + 1), np.zeros(self.size)
            delta = np.zeros(self.size + 1)

            v[0] = 3 * dx[0]**2
            
            u[0] = dx[0]**3
            u[1] = 2 * dx[0]
            
            w[0] = dx[0]**2

            delta[0] = dy[0] - BC[0] * dx[0]
            delta[1] = BC[1] - BC[0]
        
        else:
            v = np.array([dx[i] for i in range(2, len(dx))])
            w = np.array([dx[i] for i in range(len(dx) - 2)])
            u = np.array([2 * (dx[i] + dx[i+1]) for i in range(len(dx) - 1)])

        
            delta = np.array([(dy[i]/dx[i] * dx[i+1] + dy[i+1]/dx[i+1] * dx[i]) for i in range(len(dx) - 1)])
            delta = 3 * delta

            delta[0]  = delta[0]  - dx[1]  * BC[0]
            delta[-1] = delta[-1] - dx[-1] * BC[-1]


        beta, alpha, gamma = lu(v, u, w)
        sol = solver(beta, alpha, gamma, delta)


        if len(X) == 2:
            self.params[0][0] = sol[0]
            self.params[1][0] = sol[1]
            self.params[2][0] = BC[0]
            self.params[3][0] = Y[0]
        else:
            self.params[2][0] = BC[0]
            for i in range(1, self.size):
                self.params[2][i] = sol[i - 1]

            for i in range(self.size - 1):
                self.params[0][i] = ((self.params[2][i] + self.params[2][i + 1]) * dx[i] - 2 * dy[i]) / dx[i]**3
            self.params[0][-1] = ((self.params[2][-1] + BC[-1]) * dx[-1] - 2 * dy[-1]) / dx[-1]**3

            for i in range(self.size - 1):
                self.params[1][i] = (3 * dy[i] - (self.params[2][i + 1] + 2 * self.params[2][i]) * dx[i]) / dx[i]**2
            self.params[1][-1] = (3 * dy[-1] - (BC[-1] + 2 * self.params[2][-1]) * dx[-1]) / dx[-1]**2

            self.params[3] = np.array([Y[i] for i in range(len(X) - 1)])
            
            
    def eval(self, x: float) -> float:
        """
        Evaluate the spline value at a given x value.

        Parameters
        ------------------
        x : float
            Location at which computing the value.
        
        Returns
        ------------------
        float:
            Returns the corresponding values.

        Raises
        ------------------
        ValueError: if one of the following conditions are met:
            - the input value is out of the node domain.
        """

        if x < self.nodes[0]:
            raise ValueError
        if x > self.nodes[-1]:
            raise ValueError
        
        k = 0

        while x > self.nodes[k + 1]:
            k = k + 1

        return self.params[0][k] * (x - self.nodes[k])**3 + self.params[1][k] * (x - self.nodes[k])**2 + self.params[2][k] * (x - self.nodes[k]) + self.params[3][k]