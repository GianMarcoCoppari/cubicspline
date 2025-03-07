import unittest
import numpy as np
from ..code.tls import *

class TestForewardSubstitution(unittest.TestCase):
    # check dimensions
    def test_min_size(self):
        self.assertRaises(AssertionError, foreward, np.array([]), np.array([1]), np.array([2]))
    
    def test_diagonal_solution_size(self):
        self.assertRaises(AssertionError, foreward, np.array([1, 1, 1]), np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1, 1]))

    def test_diagonal_lower_diagonal_dimension(self):
        self.assertRaises(AssertionError, foreward, np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1]))

    #check divisions by zero
    def test_invalid_alpha(self):
        alphas = [
            np.array([0, 1, 1, 1]), 
            np.array([1, 0, 1, 1]),
            np.array([1, 1, 0, 1]),
            np.array([1, 1, 1, 0])
        ]

        for i in range(len(alphas)):
            self.assertRaises(ValueError, foreward, np.array([3, 2, 2]), alphas[i], np.array([1, 2, 3, 4]))

    # check known solution
    def test_known_solution(self):
        sol = foreward(np.array([3, 2, 2]), np.array([6, 17/2, 128/17, 111/32]), np.array([-6, 39/2, -21/2, 6]))
        x = np.array([-1, 45/17, -537/256, 435/148])

        for i in range(len(sol)):
            self.assertAlmostEqual(sol[i], x[i])

class TestBackwardSubstitution(unittest.TestCase):
    # check dimensions
    def test_min_diagonal_size(self):
        self.assertRaises(AssertionError, backward, np.array([]), np.array([1]))

    def test_size(self):
        self.assertRaises(AssertionError, backward, np.array([1, 1, 1]), np.array([1, 1, 1]))

    # check known solution
    def test_known_solution(self):
        gamma = np.array([1/2, 4/17, 17/64])
        temp = np.array([-1, 45/17, -537/256, 435/148])

        b = np.array([-197/74, 123/37, -213/74, 435/148])
        sol = backward(gamma, temp)

        for i in range(len(sol)):
            self.assertAlmostEqual(b[i], sol[i])

class TestSolver(unittest.TestCase):
    # check known solution
    def test_known_solution(self):
        alpha = np.array([6, 17/2, 128/17, 111/32])
        beta  = np.array([3, 2, 2])
        gamma = np.array([1/2, 4/17, 17/64])
        delta = np.array([-6, 39/2, -21/2, 6])

        b = np.array([-197/74, 123/37, -213/74, 435/148])
        sol = solver(beta, alpha, gamma, delta)

        for i in range(len(sol)):
            self.assertAlmostEqual(b[i], sol[i])