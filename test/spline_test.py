import unittest
import numpy as np
from ..code.spline import CubicSpline

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
            np.array([443/666, -153/148, 287/296, -805/296]),
            np.array([-197/74, 123/37, -213/74, 435/148]),
            np.array([0, 147/74, 213/74, 3]),
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
