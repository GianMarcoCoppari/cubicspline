"""
Defines the LU matrix decomposition algorithm. It is specialized for tridiagonal matrices.
"""
import numpy as np

def lu(v: np.array, u: np.array, w: np.array) -> list[np.array, np.array, np.array]:
    """ 
    Lower-Upper Matrix Factorization algorithm using numpy arrays. 
        
        Parameters
        -----------------
        v : np.array
            Lower diagonal numbers.
        u : np.array
            Main diagonal numbers.
        w : np.array
            Upper diagonal number.

        Returns
        -----------------
        list
            Returns a list of numpy arrays. The order is L-lower diagonal, L-diagonal and U-upper diagonal elements.

        Raises
        -----------------
        AssertionError: one of the following conditions are met:
            - diagonal array has size less than two
            - off-diagonal elements do not match in size
            - diagonal and off-diagonal elements do not have proper relatie size
        ZeroDivisionError: when division by 0 is met during the algorithm.
    """

    if len(u) < 2:
        raise AssertionError
    if len(v) != len(w):
        raise AssertionError
    if len(u) != len(v) + 1:
        raise AssertionError

    beta  = np.array(v)
    alpha = np.zeros(len(u))
    gamma = np.zeros(len(w))

    # prompt for the loop
    alpha[0] = u[0]
    if u[0] != 0:
        gamma[0] = w[0] / u[0]
    else:
        raise ZeroDivisionError

    for i in range(1, len(w)):
        alpha[i] = u[i] - beta[i - 1] * gamma[i - 1]
        
        if alpha[i] != 0:
            gamma[i] = w[i] / alpha[i]        
        else:
            raise ZeroDivisionError
        
    alpha[len(u) - 1] = u[len(u) - 1] - beta[len(u) - 2] * gamma[len(u) - 2]


    return [beta, alpha, gamma]