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


    pars = np.array([
        u,
        np.concatenate(([0], v)),
        np.concatenate(([0], w))
    ], dtype = np.float64)
    
    for i in range(1, len(u)):
        pars[0, i] = pars[0, i] * pars[0, i-1] - pars[1, i] * pars[2, i]
        if not pars[0, i-1] == 0:
            pars[::2, i] = pars[::2, i] / pars[0, i-1]
        else:
            raise ZeroDivisionError

    return [pars[1, 1:], pars[0], pars[2, 1:]]