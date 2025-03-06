import numpy as np

def lu(v: np.array, u: np.array, w: np.array) -> list[np.array, np.array, np.array]:
    if len(u) < 2:
        raise AssertionError
    if len(v) != len(w):
        raise AssertionError
    if len(u) != len(v) + 1:
        raise AssertionError

    beta  = np.array(v)
    alpha = np.zeros(len(u))
    gamma = np.zeros(len(w))

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