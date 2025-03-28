"""
Defines the foreward-backward algorithm to solve tridiagonal linear systems.
"""

import numpy as np

class MinSizeException(Exception):
    pass
class RelativeSizeException(Exception):
    pass

def backward(gamma: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """ 
        Backward substitution algorithm using numpy arrays. Specialized algorithm for unitriangular upper matrices with the only non-zero elements in the adjacent off-diagonal elements.
        
        Parameters
        -----------------
        gamma : np.array
            Upper diagonal numbers.
        temp : np.array
            Number vector.

        Returns
        -----------------
        np.array
            Returns a numpy array containing the solution.

        Raises
        -----------------
        - MinSizeException: if the main diagonal has less than two elements.
        - RelativeSizeException: if the main diagonal and the upper diagonal do not have the correct relative number of elements.
    """

    if len(temp) < 2:
        raise MinSizeException("Main diagonal has less than two elements.")
    if len(temp) != len(gamma) + 1:
        raise RelativeSizeException("Main diagonal and upper diagonal do not have the correct relative number of elements.")
    

    sol = temp[-1:]
    for i in range(1, len(temp)):
        sol = np.concatenate((sol, [temp[len(temp) - 1 - i] - sol[-1] * gamma[len(temp) - 1 - i]]))

    return np.flip(sol)

def forward(beta: np.ndarray, alpha: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """ 
        Foreward substitution algorithm using numpy arrays. Specialized algorithm for lower triangula matrices with the only non-zero elements in the main diagonal and its lower djacent off-diagonal.
        
        Parameters
        -----------------
        beta : np.array
            Lower diagonal numbers.
        alpha : np.array
            Diagonal numbers.
        delta : np.array
            Number vector.

        Returns
        -----------------
        np.array
            Returns a numpy array containing the solution.

        Raises
        -----------------
        - MinSizeException: if the main diagonal has less than two elements.
        - RelativeSizeException: either when the main diagonal and the known values arrays do not have the same number of elements or when the main diagonal and the lower diagonal do not have the correct relative number of elements.
        - ValueError: if alpha contains any null value.
    """

    if len(alpha) < 2:
        raise MinSizeException("Main diagonal has less than two elements.")
    if len(alpha) != len(delta):
        raise RelativeSizeException("Main diagonal and known values have different size.")
    if len(alpha) != len(beta) + 1:
        raise RelativeSizeException("Main diagonal and lower diagonal do not have the correct relative size.")
    if len(alpha[alpha == 0]) != 0:
        raise ValueError("Main diagonal contains one or more null element.")

    temp = np.array([delta[0] / alpha[0]])
    
    for i in range(1, len(alpha)):
        temp = np.concatenate((temp, [(delta[i] - temp[i - 1] * beta[i - 1]) / alpha[i]]))
        
    return temp

def solver(beta: np.ndarray, alpha: np.ndarray, gamma: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """ 
        Foreward substitution algorithm using numpy arrays. Specialized algorithm for tridiagonal linear systems.
        
        Parameters
        -----------------
        beta : np.array
            Lower diagonal numbers.
        alpha : np.array
            Diagonal numbers.
        gamma : np.array
            Upper diagonal numbers.
        delta : np.array
            Number vector.

        Returns
        -----------------
        np.array
            Returns a numpy array containing the solution.
    """
    return backward(gamma, forward(beta, alpha, delta))