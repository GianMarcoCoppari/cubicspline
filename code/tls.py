import numpy as np

def backward(gamma: np.array, temp: np.array) -> np.array:
    if len(temp) < 2:
        raise AssertionError
    if len(temp) != len(gamma) + 1:
        raise AssertionError
    

    sol = np.zeros(len(temp))
    sol[len(temp) - 1] = temp[len(temp) - 1]

    for i in range(len(temp) - 2, -1, -1):
        sol[i] = temp[i] - gamma[i] * sol[i + 1]

    return sol

def foreward(beta: np.array, alpha: np.array, delta: np.array) -> np.array:
    if len(alpha) < 2 or len(delta) < 2:
        raise AssertionError
    if len(alpha) != len(delta):
        raise AssertionError
    if len(alpha) != len(beta) + 1:
        raise AssertionError
    if len(alpha[alpha == 0]) != 0:
        raise ValueError

    temp = np.zeros(len(alpha))
    temp[0] = delta[0] / alpha[0]
    
    for i in range(1, len(alpha)):
        temp[i] = (delta[i] - temp[i - 1] * beta[i - 1]) / alpha[i]
        
    return temp


def solver(beta: np.array, alpha: np.array, gamma: np.array, delta: np.array) -> np.array:
    return backward(gamma, foreward(beta, alpha, delta))