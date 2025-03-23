"""
Defines the foreward-backward algorithm to solve tridiagonal linear systems.
"""
import numpy as np

def backward(gamma: np.array, temp: np.array) -> np.array:
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
        AssertionError: if one of the following conditions are met:
            - system size is less than two.
            - diagonal and off-diagonal elements do not have proper relatie size
    """

    if len(temp) < 2:
        raise AssertionError
    if len(temp) != len(gamma) + 1:
        raise AssertionError
    

    sol = temp[-1:]
    for i in range(1, len(temp)):
        sol = np.concatenate((sol, [temp[2 - i] - sol[-1] * gamma[2 - i]]))

    return np.flip(sol)

def forward(beta: np.array, alpha: np.array, delta: np.array) -> np.array:
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
        AssertionError: if one of the following conditions are met:
            - system size is less than two.
            - diagonal and off-diagonal elements do not have proper relatie size
        ValueError: if alpha contains any null value.
    """

    if len(alpha) < 2 or len(delta) < 2:
        raise AssertionError
    if len(alpha) != len(delta):
        raise AssertionError
    if len(alpha) != len(beta) + 1:
        raise AssertionError
    if len(alpha[alpha == 0]) != 0:
        raise ValueError

    temp = np.array([delta[0] / alpha[0]])
    
    for i in range(1, len(alpha)):
        temp = np.concatenate((temp, [(delta[i] - temp[i - 1] * beta[i - 1]) / alpha[i]]))
        
    return temp


def solver(beta: np.array, alpha: np.array, gamma: np.array, delta: np.array) -> np.array:
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