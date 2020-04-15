import numpy as np

from mnistclassifier.activation_functions import sigmoid


class Network:
    """The actual network object"""

    def __init__(self, sizes, weights_and_biases="default"):
        """The constructor takes as an argument a list of integers the
         number of elements gives the amount of layer and the integer gives
          the amount of neurons in that layer"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        if weights_and_biases == "default":
            self.biases, self.weights = self.initialize_default_weights()


    def initialize_default_weights(self):
        """Initialize the weights as Gaussian random variables with mean 0 and
        and standard deviation 1 divided by the square root of number of
        connections to the neuron"""
        biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in
                   zip(self.sizes[:-1],
                       self.sizes[1:])]
        return biases, weights

    def initialize_large_weights(self):
        """Initialize the weights as Gaussian random variables with mean 0 and
        and standard deviation 1"""
        biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in
                   zip(self.sizes[:-1],
                       self.sizes[1:])]
        return biases, weights


    def feedforward(self, a):
        """Returns the output if a is the input """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
