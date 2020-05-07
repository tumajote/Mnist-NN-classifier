import numpy as np

from mnistclassifier.activation_functions import sigmoid
from mnistclassifier.cost_functions import CrossEntropyCost


class Network:
    """The actual data structure which maintains the weights and biases of the
    network and does the classification"""

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The Network constructor takes as an argument a list of integers the
         number of elements gives the amount of layers and the integer gives
          the amount of neurons in that layer"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost

        """Initialize the biases as Gaussian random variables with mean 0 and
        and standard deviation 1 and weights as Gaussian random variables with
        mean 0 and and standard deviation 1 divided by their square roots"""
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in
                        zip(self.sizes[:-1],
                            self.sizes[1:])]

    def feedforward(self, a):
        """Returns the output the network if a is the input. a must be a vector
        with the length of 784 which is the amount of pixels in a picture in
        the MNIST dataset"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
