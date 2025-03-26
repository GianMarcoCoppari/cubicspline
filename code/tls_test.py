import unittest
import numpy as np
from lu import lu
import tls

class TestForewardSubstitution(unittest.TestCase):
    # check dimensions
    def test_min_size(self):
        """
            Test that the system has the minimum number of equations. 
            If the size is less than two raises an exception.
        """
        beta = np.array([])
        alpha = np.array([1])
        delta = np.array([])

        with self.assertRaises(AssertionError):
            tls.forward(beta, alpha, delta)
    
    def test_diagonal_dimension(self):
        """
            Test that the diagonal array has the same number of elements of the known values.
            If they are different an exception is raised.
        """
        beta = np.array([1])
        alpha = np.array([1, 2])
        delta = np.array([2, 2, 2])

        with self.assertRaises(AssertionError):
            tls.forward(beta, alpha, delta)
    
    def test_diagonal_off_diagonal_size(self):
        """ 
            Test that the main diagonal and the off-diagonal arrays have the proper relative size.
            If they do not have the same relative size an exception is raised.
        """
        beta  = np.array([2, 2])
        alpha = np.array([2, 2])
        delta = np.array([2, 2])

        with self.assertRaises(AssertionError):
            tls.forward(beta, alpha, delta)

    #check divisions by zero
    def test_invalid_alpha(self):
        """
            Null values in the main diagonal will end in division by zero in the algorithm.
            If they are met while scanning the input array an exception is raised.
        """
        alpha = np.array([0, 1, 1, 1])
        beta = np.array([3, 2, 2])
        delta = np.array([1, 2, 3, 4])

        with self.assertRaises(ValueError):
            tls.forward(beta, alpha, delta)

    # check known solution
    def test_known_solution(self):
        """
            Test the correctness of a known problem.
        """
        beta = np.array([2, 2])
        alpha = np.array([10, 37/5, 276/37])
        delta = np.array([57/2, 33, -6])

        x = tls.forward(beta, alpha, delta)
        sol = np.array([57/20, 273/74, -165/92])

        self.assertTrue(np.allclose(x, sol))

    def test_upper_triangular_solution(self):
        """
            Special case: upper triangular problem.
            The given set of array correspond to an upper triangular matrix. 
            The factorization is made up of a diagonal matrix and an upper unitriangular one. 
            The computation of the solution should present no problem if the input values are correct.
        """
        v = np.array([0, 0, 0], dtype = np.float64)
        u = np.array([1, 4, -3, 6], dtype = np.float64)
        w = np.array([-2, -7, 0], dtype = np.float64)
        delta = np.array([3, 0, 1, -4], np.float64)
        
        sol = np.array([3, 0, -1/3, -2/3], dtype = np.float64)
        temp = lu(v, u, w)
        x = tls.forward(temp[0], temp[1], delta)

        self.assertTrue(np.allclose(x, sol))

class TestBackwardSubstitution(unittest.TestCase):
    # check dimensions
    def test_min_diagonal_size(self):
        """
            Test that the main diagonal has at least two elements.
            If not an exception is raised.
        """
        gamma = np.array([])
        temp = np.array([1])

        with self.assertRaises(AssertionError):
            tls.backward(gamma, temp)

    def test_size(self):
        """
            Test that the two arrays have the proper relative dimensions.
            If array 'gamma' has not exactly the number 
            of elements of 'temp' - 1 an exception is raised.
        """
        gamma = np.array([1, 1, 1])
        temp = np.array([1, 1, 1])
        
        with self.assertRaises(AssertionError):
            tls.backward(gamma, temp)

    # check known solution
    def test_known_solution(self):
        """
            Test the correctness of the algorithm computin the solution of a known problem.
        """
        gamma = np.array([3/10, 10/37])
        temp = np.array([57/20, 273/74, -165/92])

        sol = np.array([147/92, 96/23, -165/92])
        x = tls.backward(gamma, temp)

        self.assertTrue(np.allclose(sol, x))
    
class TestSolver(unittest.TestCase):
    # check known solution
    def test_known_solution(self):
        """
            Test the correctness of the algorithm computin the solution of a known problem,
            when valid inputs are pass to the function.
        """
        alpha = np.array([10, 37/5, 276/37])
        beta  = np.array([2, 2])
        gamma = np.array([3/10, 10/37])
        delta = np.array([57/2, 33, -6])

        sol = np.array([147/92, 96/23, -165/92])
        x = tls.solver(beta, alpha, gamma, delta)

        self.assertTrue(np.allclose(x, sol))

    def test_lower_triangular_solution(self):
        """
            Test the particular case of lower triangular maxtrix, 
            computing the solution of a known problem.
        """
        v = np.array([-3, 5, 2], dtype = np.float64)
        u = np.array([1, -4, 3, -10], dtype = np.float64)
        w = np.array([0, 0, 0], dtype = np.float64)

        beta, alpha, gamma = lu(v, u, w)
        delta = np.array([-3, 0.0000000001, 0, 1], dtype = np.float64)

        sol = np.array([-3, 89999999999/40000000000, -89999999999/24000000000, -101999999999/120000000000], dtype = np.float64)
        x = tls.solver(beta, alpha, gamma, delta)

        self.assertTrue(np.allclose(x, sol))

    def test_upper_triangular_solution(self):
        """
            Test the particular case of upper triangular matrices,
            computing the solution to a known problem.
        """
        v = np.array([0, 0, 0], dtype = np.float64)
        u = np.array([1, 4, -3, 6], dtype = np.float64)
        w = np.array([-2, -7, 0], dtype = np.float64)

        beta, alpha, gamma = lu(v, u, w)

        delta = np.array([3, 0, 1, -4])
        
        x = tls.solver(beta, alpha, gamma, delta)
        sol = np.array([11/6, -7/12, -1/3, -2/3])

        self.assertTrue(np.allclose(x, sol))

    def test_diagonal_solution(self):
        """
            Test the particular case of diagonal matrices, 
            computing the solution to a known problem.
        """
        v = np.array([0, 0, 0, 0], dtype = np.float64)
        u = np.array([1, -5, 3, -0.2, 0.00075], dtype = np.float64)
        w = np.array([0, 0, 0, 0], dtype = np.float64)

        beta, alpha, gamma = lu(v, u, w)
        delta = np.array([4, -8, 2, -0.3, 0.1], dtype = np.float64)

        sol = delta / u
        x = tls.solver(beta, alpha, gamma, delta)

        self.assertTrue(np.allclose(x, sol))

    def test_lower_diagonal_solver(self):
        """
            Test the correctness of the algorithm comparing the solutions 
            to the same input problem, obtained with different strategies.
            Since a lower diagonal matrix is factorized into the same lower diagonal matrix
            and the identity matrix (correspondign to gamma = 0), the problem can be solved
            using only the forward substitution.
        """
        v = np.array([-3, 5, 2], dtype = np.float64)
        u = np.array([1, -4, 3, -10], dtype = np.float64)
        w = np.array([0, 0, 0], dtype = np.float64)

        beta, alpha, gamma = lu(v, u, w)
        delta = np.array([-3, 0.0000000001, 0, 1], dtype = np.float64)

        xsolver = tls.solver(beta, alpha, gamma, delta)
        xforward = tls.forward(beta, alpha, delta)

        self.assertTrue(np.allclose(xsolver, xforward))

    def test_upper_diagonal_solver(self):
        """
            Test the correctness of the algorithm comparing the solutions to the same input
            problem, but obtained with different strategies.
        """
        v = np.array([0, 0, 0], dtype = np.float64)
        u = np.array([1, 4, -3, 6], dtype = np.float64)
        w = np.array([-2, -7, 0], dtype = np.float64)

        beta, alpha, gamma = lu(v, u, w)
        delta = np.array([3, 0, 1, -4])
        
        xsolver = tls.solver(beta, alpha, gamma, delta)
        xbackward = tls.backward(gamma, delta/alpha)
        
        self.assertTrue(np.allclose(xsolver, xbackward))