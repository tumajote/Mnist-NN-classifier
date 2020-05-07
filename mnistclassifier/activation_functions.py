import numpy as np


def sigmoid(z):
    """Returns the sigmoid function"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Returns the derivative of the sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))
