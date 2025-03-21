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
        Numpy array of two elements containing first derivatives at first and last node, respectively.
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
            Numpy array of two elements containing first derivatives at first and last node, respectively.

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
            np.zeros(self.size)
        ]
        
        print(self.params)

        dx = np.array([X[i + 1] - X[i] for i in range(self.size)])
        dy = np.array([Y[i + 1] - Y[i] for i in range(self.size)])
        

        if len(X) == 2:
            self.params = self.__two_point_spline(dx, dy, BC, Y)
        else:
            self.params = self.__multiple_point_spline(dx, dy, BC, Y)
            
            
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
    

    def __two_point_spline(self, dx: np.array, dy: np.array, BC: np.array, Y: np.array) -> list[np.array, np.array, np.array, np.array]:
        v, u, w = np.zeros(len(dx)), np.zeros(len(dx) + 1), np.zeros(len(dx))
        delta = np.zeros(len(dx) + 1)

        v[0] = 3 * dx[0]**2
            
        u[0] = dx[0]**3
        u[1] = 2 * dx[0]
            
        w[0] = dx[0]**2

        delta[0] = dy[0] - BC[0] * dx[0]
        delta[1] = BC[1] - BC[0]

        beta, alpha, gamma = lu(v, u, w)
        sol = solver(beta, alpha, gamma, delta)

        return [np.array([sol[0]]),
                np.array([sol[1]]),
                np.array([BC[0]]),
                np.array([Y[0]])]
    def __multiple_point_spline(self, dx: np.array, dy: np.array, BC: np.array, Y: np.array) -> list[np.array, np.array, np.array, np.array]:
        v = np.array([dx[i] for i in range(2, len(dx))])
        w = np.array([dx[i] for i in range(len(dx) - 2)])
        u = np.array([2 * (dx[i] + dx[i+1]) for i in range(len(dx) - 1)])

        print("v:", v)
        print("u:", u)
        print("w:", w)
        
        delta = np.array([(dy[i]/dx[i] * dx[i+1] + dy[i+1]/dx[i+1] * dx[i]) for i in range(len(dx) - 1)])
        delta = 3 * delta

        delta[0]  = delta[0]  - dx[1]  * BC[0]
        delta[-1] = delta[-1] - dx[-1] * BC[-1]

        beta, alpha, gamma = lu(v, u, w)
        sol = solver(beta, alpha, gamma, delta)

        pars = [
            np.zeros(len(dx)), # paramteri a
            np.zeros(len(dx)), # parametri b
            np.zeros(len(dx)), # parametri c
            np.zeros(len(dx))  # paramteri d
        ]

        pars[2] = np.concatenate((BC[:1], sol))
        
        
        for i in range(len(dx) - 1):
            pars[0][i] = ((pars[2][i] + pars[2][i + 1]) * dx[i] - 2 * dy[i]) / dx[i]**3
        pars[0][-1] = ((pars[2][-1] + BC[-1]) * dx[-1] - 2 * dy[-1]) / dx[-1]**3

        for i in range(len(dx) - 1):
            pars[1][i] = (3 * dy[i] - (pars[2][i + 1] + 2 * pars[2][i]) * dx[i]) / dx[i]**2
        pars[1][-1] = (3 * dy[-1] - (BC[-1] + 2 * pars[2][-1]) * dx[-1]) / dx[-1]**2

        pars[3] = np.array([Y[i] for i in range(len(dx))])

        return pars
