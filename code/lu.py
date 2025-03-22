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


    seed = np.array([u[0], v[0]])
    if not seed[-2] == 0:
        seed = np.concatenate((seed, w[:1]/seed[-2]))
    else:
        raise ZeroDivisionError
    
    for i in range(1, len(u) - 1):
        seed = np.concatenate((seed, u[i:i+1] - seed[-1] * seed[-2], v[i:i+1]))
        if not seed[-2] == 0:
            seed = np.concatenate((seed, w[i:i+1] / seed[-2]))
        else:
            raise ZeroDivisionError

    seed = np.concatenate((seed, u[-1:] - seed[-1] * seed[-2]))

    return [seed[1::3], seed[::3], seed[2::3]] # [beta, alpha, gamma], in the order
