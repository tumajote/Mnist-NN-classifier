import numpy as np

from mnistclassifier.activation_functions import sigmoid_prime


"""
Quadratic cost function for legacy use

class QuadraticCost:
    @staticmethod
    def function(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2
    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)
"""

class CrossEntropyCost:
    @staticmethod
    def function(a, y):
        """Returns the cost for output a and desired output y"""
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """Returns the error delta for the output layer. Z is included to make
        the method consistent with other cost functions"""
        return a - y
