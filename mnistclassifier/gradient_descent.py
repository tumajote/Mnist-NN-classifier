import random

import numpy as np

from mnistclassifier.backpropagate import backpropagate


def train_with_stochastic_gradient_descent(network, training_data, epochs,
                                           mini_batch_size,
                                           learning_rate,
                                           regularization_parameter=0.0,
                                           evaluation_data=None):
    """Divide the training dataset into mini batches and update the mini
     batches according the learning rate and regularization parameter.
      If evaluation_data is provided, the network will be evaluated against
      the evaluation dataset after each epoch"""
    if evaluation_data:
        n_data = len(evaluation_data)

    """The monitoring can be set from these variables"""
    monitor_evaluation_cost = True,
    monitor_evaluation_accuracy = True,
    monitor_training_cost = True,
    monitor_training_accuracy = True

    n = len(training_data)
    evaluation_cost, evaluation_accuracy = [], []
    training_cost, training_accuracy = [], []

    """Divide and update the mini batches"""
    for j in range(epochs):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k + mini_batch_size]
            for k in range(0, n, mini_batch_size)
        ]
        for mini_batch in mini_batches:
            update_mini_batch(network, mini_batch, learning_rate,
                              regularization_parameter, n)
        print("Epoch %s complete" % j)

        """Print the results to console and save them for plotting"""
        if monitor_training_cost:
            cost = total_cost(network, training_data, regularization_parameter)
            training_cost.append(cost)
            print("Cost on training data: {}".format(cost))
        if monitor_training_accuracy:
            single_training_accuracy = accuracy(network, training_data,
                                                convert=True)
            training_accuracy.append(single_training_accuracy)
            print("Accuracy on training data:"
                  " {} / {}".format(single_training_accuracy, n))
        if monitor_evaluation_cost and evaluation_data:
            cost = total_cost(network, evaluation_data,
                              regularization_parameter, convert=True)
            evaluation_cost.append(cost)
            print("Cost on evaluation data: {}".format(cost))
        if monitor_evaluation_accuracy and evaluation_data:
            single_training_accuracy = accuracy(network, evaluation_data)
            evaluation_accuracy.append(single_training_accuracy)
            print("Accuracy on evaluation data:"
                  " {} / {}".format(single_training_accuracy,
                                    n_data))

    return evaluation_cost, evaluation_accuracy, training_cost, \
           training_accuracy


def update_mini_batch(network, mini_batch, learning_rate,
                      regularization_parameter, n):
    """Update the network according the mini batch by applying gradient descent
     with backpropagation and L2 regularization"""
    nabla_b = [np.zeros(b.shape) for b in network.biases]
    nabla_w = [np.zeros(w.shape) for w in network.weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = backpropagate(network, x, y)
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    network.weights = [
        (1 - learning_rate * (regularization_parameter / n)) * w - (
                learning_rate / len(mini_batch)) * nw for w, nw in
        zip(network.weights, nabla_w)]
    network.biases = [b - (learning_rate / len(mini_batch)) * nb
                      for b, nb in zip(network.biases, nabla_b)]


def accuracy(network, data, convert=False):
    """Returns the number of correct results. The convert flag should be set
     to true if the dataset is the training data as the correct output is in
     a different format"""
    if convert:
        results = [(np.argmax(network.feedforward(x)), np.argmax(y))
                   for (x, y) in data]
    else:
        results = [(np.argmax(network.feedforward(x)), y)
                   for (x, y) in data]
    number_of_correct_results = sum(int(x == y) for (x, y) in results)
    return number_of_correct_results


def total_cost(network, data, regularization_parameter, convert=False):
    """Returns the total cost for the data set.  The convert flag should be set
     to true if the dataset is the training data as the correct output is in
     a different format"""

    cost = 0.0
    for x, y in data:
        a = network.feedforward(x)
        if convert:
            y = vectorized_result(y)
        cost += network.cost.function(a, y) / len(data)
    cost += 0.5 * (regularization_parameter / len(data)) * sum(
        np.linalg.norm(w) ** 2 for w in network.weights)
    return cost


def vectorized_result(y):
    """Returns the correct output as a vector where the index is the digit"""
    vector = np.zeros((10, 1))
    vector[y] = 1.0
    return vector
