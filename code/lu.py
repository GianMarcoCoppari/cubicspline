"""
Defines the LU matrix decomposition algorithm. It is specialized for tridiagonal matrices.
"""

class MinSizeException(Exception):
    pass
class OffDiagonalSizeException(Exception):
    pass
class RelativeSizeException(Exception):
    pass

import numpy as np


def lu(v: np.ndarray, u: np.ndarray, w: np.ndarray) -> list[np.ndarray]:
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
        - MinSizeException: if the array associaed with the main diagonal has less than two elements.
        - RelativeSizeException: if either one of the two off diagonal has a number of element different from the number of elements in the main diagonal minus one.
        - ZeroDivisionError: when division by 0 is met during the algorithm.
    """

    if len(u) < 2:
        raise MinSizeException("main diagonal has less than 2 elements.")
    if len(u) != len(v) + 1:
        raise RelativeSizeException("Diagonal and lower diagonal arrays do not have the correct relative size.")
    if len(u) != len(w) + 1:
        raise RelativeSizeException("Diagonal and upper diagonal arrays do not have proper relative size.")

    pars = np.array([u[1:], v, w], dtype = np.float64)
    
    # manually compute first element
    pars[0, 0] = pars[0, 0] * u[0] - pars[1, 0] * pars[2, 0] 
    
    # scaled coefficeints, il primo ed il terzo elemento della prima colonna 
    # per il primo coefficiente della colonna precedente
    if not u[0] == 0:
        pars[::2, 0] = pars[::2, 0] / u[0]
    else:
        raise ZeroDivisionError
    
    for i in range(1, len(u) - 1):
        pars[0, i] = pars[0, i] * pars[0, i - 1] - pars[1, i] * pars[2, i]
        
        if not pars[0, i - 1] == 0:
            pars[::2, i] = pars[::2, i] / pars[0, i - 1]
        else:
            raise ZeroDivisionError


    return [pars[1], np.concatenate((u[:1], pars[0])), pars[2]]