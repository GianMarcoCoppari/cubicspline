import unittest
import numpy as np
from spline import CubicSpline

class TestCubicSpline(unittest.TestCase):
    def test_X_min_size(self):
        """
            Test the minimum size of nodes. If it is less than two it raises an exception.
        """
        X  = np.array([1])
        Y  = np.array([2])
        BC = np.array([0, 0])
    
        with self.assertRaises(AssertionError):
            CubicSpline(X, Y, BC)
            
    def test_XY_same_size(self):
        """
            Test that the X and Y arrays have same length, representing x and y coordinate of nodes.
        """
        X =  np.array([1, 2])
        Y =  np.array([1])
        BC = np.array([0, 0])

        with self.assertRaises(AssertionError):
            CubicSpline(X, Y, BC)

    def test_distinct_nodes(self):
        """
            Test that the nodes are all in different positions.
            If two nodes are in the same place, i.e. they have same x coordinate
            an exception is raised.
        """
        X  = np.array([1, 3, 3, 5])
        Y  = np.array([3, 4, 5, 6])
        BC = np.array([0, 0])

        with self.assertRaises(AssertionError):
            CubicSpline(X, Y, BC)

    def test_ordered_nodes(self):
        """
            Test that the input nodes are ordered. If not an exception is raised.
        """
        X = np.array([1, 4, 3, 5])
        Y = np.array([1, 5, 7, 9])
        BC = np.array([0, 0])

        with self.assertRaises(AssertionError):
            CubicSpline(X, Y, BC)

    def test_invalid_boundary_condition_size(self):
        """
            Test if the boundary conditions array have correct number of elements.
            Raise an exception if not.
        """
        X = np.array([1, 3, 5, 7])
        Y = np.array([2, 4, 6, 8])
        BC = np.array([0])

        with self.assertRaises(AssertionError):
            CubicSpline(X, Y, BC)

    def test_known_spline(self):
        """
            Test the correctness of parameter computation with a known-solution problem.
        """

        X  = np.array([1, 4, 6, 8, 10])
        Y  = np.array([2, -4, 5, 7, 3])
        BC = np.array([0, 0])

        spline = CubicSpline(X, Y, BC)
        sol = [
            np.array([515/828, -297/368, 35/368, 203/368]),
            np.array([-233/92, 141/46, -327/184, -111/92]),
            np.array([0, 147/92, 96/23, -165/92]),
            np.array([2, -4, 5, 7])
        ]

        self.assertTrue(map(np.allclose, spline.params, sol))

    def test_two_point_spline(self):
        """
            Test particular case of two point cubic spline, 
            since the computation of parameters is defferent.
        """
        X = np.array([1, 6])
        Y = np.array([3, 1])
        BC = np.array([0, 0])
        spline = CubicSpline(X, Y, BC)

        sol = [
            np.array([4/125]),
            np.array([-6/25]),
            np.array([0]),
            np.array([3])
        ]

        self.assertTrue(map(np.allclose, spline.params, sol))

    def test_underlying_cubic_function(self):
        """ 
            Test the particular case of an underlying cubic function.
            In thi case the cubic spline must coincide with the underlying cubic polynomio.
        """
        X  = np.array([-1, 0])
        Y  = np.array([ 0, 1])
        BC = np.array([ 0, 3])
        spline = CubicSpline(X, Y, BC)

        sol = [
            np.array([1]),
            np.array([0]),
            np.array([0]),
            np.array([0])
        ]
        
        self.assertTrue(map(np.allclose, spline.params, sol))

    def test_multiple_cubic_functions(self):
        """
            Test that piecewise underlying cubic function corresponds to 
            the same global cubic polynomio.
        """
        X  = np.array([-1, 0, 1, 2])
        Y  = (X + 1)**3
        BC = 3 * (X + 1)**2

        pars = [CubicSpline(X[j : j + 2], Y[j : j + 2], BC[j : j + 2]).params for j in range(len(X) - 1)]
        solutions = [
            [
                np.array([1]),
                np.array([0]),
                np.array([0]),
                np.array([0])
            ],
            [
                np.array([1]),
                np.array([3]),
                np.array([3]),
                np.array([1])
            ],
            [
                np.array([1]),
                np.array([6]),
                np.array([12]),
                np.array([8])
            ]
        ]

        self.assertTrue(map(np.allclose, pars, solutions))

    def test_left_edge_derivative(self):
        """
            Test that the computed first derivative coincides with the one given
            by the left boundary condition.
        """
        X = np.array([1, 4, 6, 8, 10])
        Y = np.array([2, -4, 5, 7, 3])
        BC = np.array([0, 0])

        spline = CubicSpline(X, Y, BC)
        self.assertAlmostEqual(spline.params[2][0], BC[0])

    def test_right_edge_derivative(self):
        """
            Test that the computed first derivative at right edge of the domain 
            is equal to the one provided by the second boundary condition.
        """
        X = np.array([1, 4, 6, 8, 10])
        Y = np.array([2, -4, 5, 7, 3])
        BC = np.array([0, 0])

        spline = CubicSpline(X, Y, BC)
        a, b, c, d = spline.params
        self.assertAlmostEqual(3 * a[-1] * (X[-1] - X[-2])**2 + 2 * b[-1] * (X[-1] - X[-2]) + c[-1], BC[1])

class TestEvalFunction(unittest.TestCase):
    def test_invalid_lower_input(self):
        X = np.array([1, 4, 6, 8, 10])
        Y = np.array([2, -4, 5, 7, 3])
        BC = np.array([0, 0])

        spline = CubicSpline(X, Y, BC)
        self.assertRaises(ValueError, spline.eval, 0.)

    def test_invalid_upper_input(self):
        X = np.array([1, 4, 6, 8, 10])
        Y = np.array([2, -4, 5, 7, 3])
        BC = np.array([0, 0])

        spline = CubicSpline(X, Y, BC)
        self.assertRaises(ValueError, spline.eval, 11.)

    def test_left_edge(self):
        X = np.array([1, 4, 6, 8, 10])
        Y = np.array([2, -4, 5, 7, 3])
        BC = np.array([0, 0])

        spline = CubicSpline(X, Y, BC)

        self.assertAlmostEqual(spline.eval(X[0]), Y[0])

    def test_right_edge(self):
        X = np.array([1, 4, 6, 8, 10])
        Y = np.array([2, -4, 5, 7, 3])
        BC = np.array([0, 0])

        spline = CubicSpline(X, Y, BC)

        self.assertAlmostEqual(spline.eval(X[-1]), Y[-1])
    
    def test_nodes_from_sides(self):
        X = np.array([1, 4, 6, 8, 10])
        Y = np.array([2, -4, 5, 7, 3])
        BC = np.array([0, 0])

        spline = CubicSpline(X, Y, BC)

        for i in range(1, len(X) - 2):
            self.assertAlmostEqual(spline.eval(X[i]), 
                                   spline.params[0][i] * (X[i] - spline.nodes[i])**3 + spline.params[1][i] * (X[i] - spline.nodes[i])**2 + spline.params[2][i] * (X[i] - spline.nodes[i]) + spline.params[3][i])

    def test_inner_node(self):
        X = np.array([1, 4, 6, 8, 10])
        Y = np.array([2, -4, 5, 7, 3])
        BC = np.array([0, 0])

        spline = CubicSpline(X, Y, BC)

        self.assertAlmostEqual(spline.eval(X[3]), Y[3])

    def test_mid_point(self):
        X = np.array([1, 4, 6, 8, 10])
        Y = np.array([2, -4, 5, 7, 3])
        BC = np.array([0, 0])

        spline = CubicSpline(X, Y, BC)

        for i in range(len(X) - 1):
            midpoint = 0.5 * (X[i] + X[i + 1])
            y = spline.params[0][i] * (midpoint - spline.nodes[i])**3 + spline.params[1][i] * (midpoint - spline.nodes[i])**2 + spline.params[2][i] * (midpoint - spline.nodes[i]) + spline.params[3][i]
            
            self.assertAlmostEqual(spline.eval(midpoint), y)