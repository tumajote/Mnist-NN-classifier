import numpy as np

from mnistclassifier.activation_functions import sigmoid
from mnistclassifier.cost_functions import CrossEntropyCost


class Network:
    """The actual network object"""

    def __init__(self, sizes, cost=CrossEntropyCost,
                 weights_and_biases="default"):
        """The constructor takes as an argument a list of integers the
         number of elements gives the amount of layer and the integer gives
          the amount of neurons in that layer"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost

        """Initialize the weights as Gaussian random variables with mean 0 and
        and standard deviation 1"""
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in
                   zip(self.sizes[:-1],
                       self.sizes[1:])]

    def feedforward(self, a):
        """Returns the output if a is the input """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
