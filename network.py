import numpy as np

from activation_functions import sigmoid


class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Returns the output of the network given a is the input """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)


