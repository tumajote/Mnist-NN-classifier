import numpy as np

from mnistclassifier.activation_functions import sigmoid, sigmoid_prime


def backpropagate(network, x, y):
    """Calculates the gradient for the cost function"""
    nabla_b = [np.zeros(b.shape) for b in network.biases]
    nabla_w = [np.zeros(w.shape) for w in network.weights]

    activation = x
    activations = [x]
    zs = []
    """Forward"""
    for b, w in zip(network.biases, network.weights):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
    """Backward"""
    delta = network.cost.delta(zs[-1], activations[-1], y)
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    for i in range(2, network.num_layers):
        z = zs[-i]
        sp = sigmoid_prime(z)
        delta = np.dot(network.weights[-i + 1].transpose(), delta) * sp
        nabla_b[-i] = delta
        nabla_w[-i] = np.dot(delta, activations[-i - 1].transpose())
    return nabla_b, nabla_w
