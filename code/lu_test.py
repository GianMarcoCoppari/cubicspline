import unittest
import numpy as np
from lu import *

class TestDecompositionLU(unittest.TestCase):
    # dimensional checks on input values
    def test_min_diagonal_size(self):
        """ 
            Check that the main diagonal has at least the minimum number of elements. 
            If the number of elements in the main diagonal is less than two, 
            it raises an exception.
        """

        v = np.array([])
        u = np.array([1])
        w = np.array([])

        with self.assertRaises(MinSizeException):
            lu(v, u, w)


    def test_diagonal_upper_diagonal_relative_size(self):
        """
            Check if the main diagonal and the upper diagonal have the relative right number of elements.
            If the off diagonal has a numer of element different from 'n-1', 
            where 'n' is the number of elements in the main diagonal it should raise and exception.
        """

        v = np.array([1, 1])
        u = np.array([1, 1, 1])
        w = np.array([1])
        
        with self.assertRaises(RelativeSizeException):
            lu(v, u, w)


    def test_diagonal_lower_diagonal_relative_size(self):
        """
            Check if the main diagonal and the lower diagonal have the relative right number of elements.
            If the off diagonal has a numer of element different from 'n-1', 
            where 'n' is the number of elements in the main diagonal it should raise and exception.
        """

        v = np.array([1, 1, 1])
        u = np.array([1, 1, 1])
        w = np.array([1, 1])
        
        with self.assertRaises(RelativeSizeException):
            lu(v, u, w)

    # test known problems
    def test_known_solution(self):
        """
            Compute an actual solved problems and check the correctness of the solution.
            Fails if the computed solution is different from the known one.
        """

        v = np.array([2, 2])
        u = np.array([10, 8, 8])
        w = np.array([3, 2])
        
        diags = lu(v, u, w)
        sol = [
            np.array([2, 2]), 
            np.array([10, 37/5, 276/37]), 
            np.array([3/10, 10/37])
        ]

        self.assertTrue(map(np.array_equal, sol, diags))

    def test_large_and_small_numbers(self):
        """
            Check computation stability with simultaneous handling of large and small numbers.
        """
        v = np.array([0.000002, 0.0001])
        u = np.array([200, .00025, 5])
        w = np.array([3.75, 0.05])

        diags = lu(v, u, w)
        sol = [
            np.array([0.000002, 0.0001]),
            np.array([200, 199.9999999625, 199.9999999375]),
            np.array([0.01875, 0.000250000000046875])
        ]

        self.assertTrue(map(np.array_equal, diags, sol))

    def test_negative_numbers(self):
        """
            Test the behaviour with negative numbers.
            Should compute the correct solution, unless an exception is raised.
        """
        
        v = np.array([-1, 2], dtype = np.float64)
        u = np.array([2, 1, 4], dtype = np.float64)
        w = np.array([2, -1], dtype = np.float64)

        diags = lu(v, u, w)
        sol = [
            np.array([1, -2], dtype = np.float64),
            np.array([2, 2, 5], dtype = np.float64),
            np.array([1, -1/2], dtype = np.float64)
        ]

        self.assertTrue(map(np.array_equal, diags, sol))

    def test_null_lower_diagonal(self):
        """
            Factorize an upper triangular matrix. 
            According to the unitriangular hypothesis of the upper triagular matrix 
            the result should be a diagonal matrix (beta = 0, only alpha != 0) and the 
            upper unitringular matrix
        """

        v = np.array([0, 0, 0],     dtype = np.float64)
        u = np.array([2, -1, 7, 4], dtype = np.float64)
        w = np.array([-1, -1, -5],  dtype = np.float64)

        sol = [
            np.array([0, 0, 0],        dtype = np.float64),
            np.array([2, 1, 7, 4],     dtype = np.float64),
            np.array([-1/2, -1, -5/7], dtype = np.float64)
        ]
        diags = lu(v, u, w)

        self.assertTrue(map(np.array_equal, sol, diags))

    def test_null_upper_diagonal(self):
        """
            Check the LU factorization in case the upper diagonal is null.
            In this case we expect gamma = 0, with the upper unitrangular matrix becoming the identity matrix.
            In this case the lower triangular is already facorized.
        """

        v = np.array([10000000, 3, -3])
        u = np.array([3, -1, 5, 0.00000005])
        w = np.array([0, 0, 0])

        sol = [
            np.array([10000000, 3, -3]),
            np.array([3, -1, 5, 0.00000005]),
            np.array([0, 0, 0])
        ]
        diags = lu(v, u, w)

        self.assertTrue(map(np.array_equal, sol, diags))

    # division by zero 
    def test_first_diagonal_element_zero(self):
        """
            The input diagonals contain a 0 in the first element of the main diagonal.
            It should raise a ZeroDivionError when computing the first element of the resulting upper-diagonal.
        """
        v = np.array([3, 2, 2])
        u = np.array([0, 10, 8, 4])
        w = np.array([3, 2, 2])

        with self.assertRaises(ZeroDivisionError):
            lu(v, u, w)

    def test_second_diagonal_element_zero(self):
        """
            Division by 0 is met at second entry, 
            it should raise an error when the invalid division is detected.
        """
        v = np.array([1, -2])
        u = np.array([2, 1, 4])
        w = np.array([2, -1])

        with self.assertRaises(ZeroDivisionError):
            lu(v, u, w)

unittest.main()