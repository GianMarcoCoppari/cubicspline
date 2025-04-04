"""
Defines the CubicSpline class.
"""

class MinSizeException(Exception):
    pass
class RelativeSizeException(Exception):
    pass
class UniqueNodeException(Exception):
    pass
class UnorderedSetException(Exception):
    pass
class BoundaryConditionException(Exception):
    pass

import numpy as np
from lu import lu
from tls import solver

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

    def __init__(self, X: np.ndarray, Y: np.ndarray, BC: np.ndarray):
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
            raise MinSizeException("Less than two nodes proveided.")
        if len(X) != len(Y):
            raise RelativeSizeException("X and Y do not ha same size.")
        if len(X) != len(np.unique(X)):
            raise UniqueNodeException("X does not contain unique elements.")
        if not np.all(np.diff(X) > 0):
            raise UnorderedSetException("X elements are unordered.")
        if len(BC) != 2:
            raise BoundaryConditionException("Exactly two boundary conditions are required.")

        self.nodes = X
        self.size = len(X) - 1
        self.params = []


        dx = np.array([X[i + 1] - X[i] for i in range(self.size)])
        dy = np.array([Y[i + 1] - Y[i] for i in range(self.size)])
        

        if len(X) == 2:
            self.params = self.__two_point_spline(dx, dy, BC, Y)
        else:
            self.params = self.__multiple_point_spline(dx, dy, BC, Y)
            
            
    def eval(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the spline value at a given x value.

        Parameters
        ------------------
        x : np.array
            Set of points at which compute the spline.
        
        Returns
        ------------------
        np.array:
            Returns the corresponding values.

        Raises
        ------------------
        ValueError: if one of the following conditions are met:
            - the input value is out of the node domain.
        """

        if np.min(x) < self.nodes[0]:
            raise ValueError
        if np.max(x) > self.nodes[-1]:
            raise ValueError
        
        k = np.searchsorted(self.nodes, x, 'right') - 1 # get the right interval index
        k = np.clip(k, 0, len(self.nodes) - 2)          # squeeze the indices in the right range

        dx = x - self.nodes[k]
        return self.params[0][k] * dx**3 + self.params[1][k] * dx**2 + self.params[2][k] * dx + self.params[3][k]

        

    def __two_point_spline(self, dx: np.ndarray, dy: np.ndarray, BC: np.ndarray, Y: np.ndarray) -> list[np.ndarray]:
        """
        Private method implementing the spline parameters' computation for the 
        special case of two point spline.
        
        Parameters
        --------------
        dx : np.array
            Numpy array containing x displacements between consecutive nodes.
        dy : np.array
            Numpy array containing y displacements between consecutive nodes.
        BC : np.array
            Numpy array of two elements containing first derivatives at first and last node, respectively.
        Y  : np.array
            Numpy array containing y values of spline nodes.
        Returns
        ---------------
        CubicSpline
            Returns the list of parameters of the spline.
        """
        
        v = 3 * dx**2
        u = np.array([dx[0]**3, 2 * dx[0]])
        w = dx**2

        delta = np.array([dy[0] - BC[0] * dx[0], BC[1] - BC[0]])

        beta, alpha, gamma = lu(v, u, w)
        sol = solver(beta, alpha, gamma, delta)

        return [sol[:1], sol[1:], BC[:1], Y[:len(dx)]]
    def __multiple_point_spline(self, dx: np.ndarray, dy: np.ndarray, BC: np.ndarray, Y: np.ndarray) -> list[np.ndarray]:
        """
        Private method implementing the spline parameters' computation for the 
        general case of more than two point spline.
        
        Parameters
        --------------
        dx : np.array
            Numpy array containing x displacements between consecutive nodes.
        dy : np.array
            Numpy array containing y displacements between consecutive nodes.
        BC : np.array
            Numpy array of two elements containing first derivatives at first and last node, respectively.
        Y  : np.array
            Numpy array containing y values of spline nodes.
        Returns
        ---------------
        CubicSpline
            Returns the list of parameters of the spline.
        """

        v = dx[2:]
        w = dx[:-2]
        u = 2 * (dx[:-1] + dx[1:])

        
        delta = 3 * (dy[:-1]/dx[:-1] * dx[1:] + dy[1:]/dx[1:] * dx[:-1])
        delta[0]  = delta[0]  - dx[1]  * BC[0]
        delta[-1] = delta[-1] - dx[-1] * BC[-1]

        beta, alpha, gamma = lu(v, u, w)
        sol = solver(beta, alpha, gamma, delta)

        
        c = np.concat((BC[:1], sol))
        next = np.concat((sol, BC[:-1]))

        a = ((c + next) * dx - 2 * dy)/dx**3
        b = (3 * dy - (next + 2 * c) * dx)/dx**2
        d = Y[:len(dx)]

        return [a, b, c, d]
