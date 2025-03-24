import unittest
import numpy as np
from .spline import CubicSpline

class TestCubicSpline(unittest.TestCase):
    def test_X_min_size(self):
        X = [np.array([]), np.array([1])]
        Y = [np.array([]), np.array([2])]

        for i in range(len(X)):
            self.assertRaises(AssertionError, CubicSpline.__init__, CubicSpline, X[i], Y[i], np.array([0, 0]))
            
    def test_XY_same_size_1(self):
        X = np.array([1, 2])
        Y = np.array([1])

        self.assertRaises(AssertionError, CubicSpline.__init__, CubicSpline, X, Y, np.array([0, 0]))

    def test_XY_same_size_2(self):
        X = np.array([1, 2])
        Y = np.array([1, 3, 5])

        self.assertRaises(AssertionError, CubicSpline.__init__, CubicSpline, X, Y, np.array([0, 0]))

    def test_distinct_nodes(self):
        X = np.array([1, 3, 3, 5])
        Y = np.array([3, 4, 5, 6])

        self.assertRaises(AssertionError, CubicSpline.__init__, CubicSpline, X, Y, np.array([0, 0]))

    def test_ordered_nodes(self):
        X = np.array([1, 4, 3, 5])
        Y = np.array([1, 5, 7, 9])

        self.assertRaises(AssertionError, CubicSpline.__init__, CubicSpline, X, Y, np.array([0, 0]))

    def test_few_boundary_condition_size(self):
        X = np.array([1, 3, 5, 7])
        Y = np.array([2, 4, 6, 8])
        BCs = [np.array([]), np.array([0]), np.array([0, 0, 0])]

        for i in range(len(BCs)):
            self.assertRaises(AssertionError, CubicSpline.__init__, CubicSpline, X, Y, BCs[i])

    def test_known_spline(self):
        X  = np.array([1, 4, 6, 8, 10])
        Y  = np.array([2, -4, 5, 7, 3])
        BC = np.array([0, 0])

        cs = CubicSpline(X, Y, BC)
        sol = [
            np.array([515/828, -297/368, 35/368, 203/368]),
            np.array([-233/92, 141/46, -327/184, -111/92]),
            np.array([0, 147/92, 96/23, -165/92]),
            np.array([2, -4, 5, 7])
        ]

        for i in range(len(cs.params)):
            for j in range(len(cs.params[0])):
                self.assertAlmostEqual(cs.params[i][j], sol[i][j])

    def test_two_point_spline(self):
        X = np.array([1, 6])
        Y = np.array([3, 1])
        BC = np.array([0, 0])
        cs = CubicSpline(X, Y, BC)

        sol = [
            np.array([4/125]),
            np.array([-6/25]),
            np.array([0]),
            np.array([3])
        ]

        for i in range(len(sol)):
            for j in range(len(cs.params[0])):
                self.assertAlmostEqual(sol[i][j], cs.params[i][j])

    def test_underlying_cubic_function(self):
        X  = np.array([-1, 0])
        Y  = np.array([ 0, 1])
        BC = np.array([ 0, 3])

        sol = [
            np.array([1]),
            np.array([0]),
            np.array([0]),
            np.array([0])
        ]
        cs = CubicSpline(X, Y, BC)

        for i in range(len(sol)):
            for p in range(len(sol[0])):
                self.assertAlmostEqual(sol[i][p], cs.params[i][p])

    def test_multiple_cubic_functions(self):
        X  = np.array([-1, 0, 1, 2])
        Y  = np.array([ 0, 1, 8, 27])
        BC = np.array([ 0, 3, 12, 27])

        splines = [CubicSpline(X[j : j + 2], Y[j : j + 2], BC[j : j + 2]) for j in range(len(X) - 1)]
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

        for i in range(len(splines)):
            for p in range(len(splines[i].params)):
                self.assertAlmostEqual(solutions[i][p][0], splines[i].params[p][0])

    def test_left_edge_derivative(self):
        X = np.array([1, 4, 6, 8, 10])
        Y = np.array([2, -4, 5, 7, 3])
        BC = np.array([0, 0])

        spline = CubicSpline(X, Y, BC)
        self.assertAlmostEqual(spline.params[2][0], BC[0])

    def test_right_edge_derivative(self):
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