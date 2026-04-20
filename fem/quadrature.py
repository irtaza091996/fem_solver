"""Quadrature rules for interval [0,1] and reference triangle, exact up to order 3."""
import numpy as np
import sys


class Quadrature:
    def __init__(self, dim, order):
        self.dim = dim
        self.order = order
        if dim == 1:
            if order <= 1:
                self.points = np.array([0.5])
                self.weights = np.array([1.0])
            else:
                if order > 3:
                    print("Requested accuracy not implemented — using order 3.")
                    self.order = 3
                self.points = np.array(
                    [(np.sqrt(3) + 1) / 2 / np.sqrt(3), (np.sqrt(3) - 1) / 2 / np.sqrt(3)]
                )
                self.weights = np.array([0.5, 0.5])
        elif dim == 2:
            if order <= 1:
                self.points = np.array([[1.0 / 3, 1.0 / 3]])
                self.weights = np.array([0.5])
            elif order <= 2:
                self.points = np.array(
                    [[1.0 / 6, 1.0 / 6], [2.0 / 3, 1.0 / 6], [1.0 / 6, 2.0 / 3]]
                )
                self.weights = np.array([1.0 / 6, 1.0 / 6, 1.0 / 6])
            else:
                if order > 3:
                    print("Requested accuracy not implemented — using order 3.")
                    self.order = 3
                self.points = np.array(
                    [
                        [1.0 / 3, 1.0 / 3],
                        [1.0 / 5, 3.0 / 5],
                        [1.0 / 5, 1.0 / 5],
                        [3.0 / 5, 1.0 / 5],
                    ]
                )
                self.weights = np.array([-27.0 / 96, 25.0 / 96, 25.0 / 96, 25.0 / 96])
        else:
            sys.exit("Chosen dimension not implemented. Aborting.")

    def integrateFunction(self, f):
        if f.__code__.co_argcount != self.dim:
            sys.exit("Function and Quadrature dimension don't match. Aborting.")
        if self.dim == 1:
            fvals = f(self.points)
        else:
            fvals = f(self.points[:, 0], self.points[:, 1])
        return np.sum(np.multiply(fvals, self.weights))
