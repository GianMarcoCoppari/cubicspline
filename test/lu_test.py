import unittest
import numpy as np
from ..code.lu import lu

class TestDecompositionLU(unittest.TestCase):
    # dimensional checks on input values
    def test_min_diagonal_size(self):
        self.assertRaises(AssertionError, lu, np.array([]), np.array([1]), np.array([]))
    
    def test_off_diagonal_equal_size(self):
        self.assertRaises(AssertionError, lu, np.array([1]), np.array([1, 2, 3]), np.array([1, 2]))
    
    def test_diagonal_off_diagonal_size(self):
        self.assertRaises(AssertionError, lu, np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1]))
    
    
    # test known problems
    def test_known_solution(self):
        v, u, w = np.array([3, 2, 2]), np.array([6, 10, 8, 4]), np.array([3, 2, 2])
        
        diags = lu(v, u, w)
        sol = [
            np.array([3, 2, 2]), 
            np.array([6, 17/2, 128/17, 111/32]), 
            np.array([1/2, 4/17, 17/64])
        ]

        for i in range(len(sol)):
            for j in range(len(sol[i])):
                self.assertAlmostEqual(sol[i][j], diags[i][j])


    # division by zero 
    def test_first_diagonal_element_zero(self):
        v, u, w = np.array([3, 2, 2]), np.array([0, 10, 8, 4]), np.array([3, 2, 2])
        self.assertRaises(ZeroDivisionError, lu, v, u, w)

    def test_second_diagonal_element_zero(self):
        v, u, w = np.array([3, 2, 2]), np.array([6, 3/2, 8, 4]), np.array([3, 2, 2])
        self.assertRaises(ZeroDivisionError, lu, v, u, w)