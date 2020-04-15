import random

import numpy as np

from mnistclassifier.backpropagate import backpropagate


def train_with_stochastic_gradient_descent(network, training_data, epochs,
                                           mini_batch_size,
                                           learning_rate, test_data=None):
    """Divide the training dataset into mini batches and update the mini
     batches according the learning rate. If test_data is provided, the network
      will be evaluated against the test dataset after each epoch"""
    if test_data:
        n_test = len(test_data)
    n = len(training_data)
    number_of_correct_results = []
    percent_of_correct_results = []
    for j in range(epochs):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k + mini_batch_size]
            for k in range(0, n, mini_batch_size)
        ]
        for mini_batch in mini_batches:
            update_mini_batch(network, mini_batch, learning_rate)
        if test_data:

            amount_of_correct_outputs = evaluate(network, test_data)
            accuracy = round(amount_of_correct_outputs / n_test, 4) * 100
            print("Epoch {0}: {1} / {2} Accuracy: {3}%".format(j,
                                                               amount_of_correct_outputs,
                                                               n_test,
                                                               accuracy))
            number_of_correct_results.append(
                (amount_of_correct_outputs, n_test))
            percent_of_correct_results.append((accuracy, 100))
        else:
            print("Epoch {0} complete".format(j))
    return number_of_correct_results, percent_of_correct_results


def update_mini_batch(network, mini_batch, learning_rate):
    """Update a mini batch by applying gradient descent with backpropagation"""
    nabla_b = [np.zeros(b.shape) for b in network.biases]
    nabla_w = [np.zeros(w.shape) for w in network.weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = backpropagate(network, x, y)
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    network.weights = [w - (learning_rate / len(mini_batch)) * nw
                       for w, nw in zip(network.weights, nabla_w)]
    network.biases = [b - (learning_rate / len(mini_batch)) * nb
                      for b, nb in zip(network.biases, nabla_b)]


def evaluate(network, test_data):
    """Returns the number of correct results"""
    test_results = [(np.argmax(network.feedforward(x)), y)
                    for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)