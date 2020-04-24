import random

import numpy as np
import pytest

from mnistclassifier.gradient_descent import \
    train_with_stochastic_gradient_descent
from mnistclassifier.network import Network


@pytest.fixture
def network():
    """Returns a Network instance with 3 layers of neurons: first 784, second
    30 and third 10 neurons in each"""
    return Network([784, 30, 10])


@pytest.fixture
def data():
    """Mock training data"""
    n = 10000
    n = int(n / 2)
    x = [np.zeros((784, 1)) for x in range(n)]
    y = [np.zeros((10, 1)) for x in range(n)]
    training_data = list(zip(x, y))
    for x, y in training_data:
        for i in range(0, 200):
            x[i] = 0.99
        y[0] = 1

    x = [np.zeros((784, 1)) for x in range(n)]
    y = [np.zeros((10, 1)) for x in range(n)]
    training_data_extra = list(zip(x, y))
    for x, y in training_data_extra:
        for i in range(300, 500):
            x[i] = 0.99
        y[3] = 1
    training_data.extend(training_data_extra)
    random.shuffle(training_data)

    """Mock test data"""
    n_test = 1000
    n_test = int(n_test / 2)
    x = [np.zeros((784, 1)) for x in range(n_test)]
    y = [0 for x in range(n_test)]
    test_data = list(zip(x, y))
    for x, y in test_data:
        for i in range(0, 200):
            x[i] = 0.99

    x = [np.zeros((784, 1)) for x in range(n_test)]
    y = [3 for x in range(n_test)]
    test_data_extra = list(zip(x, y))
    for x, y in test_data_extra:
        for i in range(300, 500):
            x[i] = 0.99
    test_data.extend(test_data_extra)
    random.shuffle(training_data)

    data = training_data, test_data

    return data


def test_training_with_stochastic_gradient_descent(network, data):
    training_data, test_data = data
    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy = train_with_stochastic_gradient_descent(
        network=network,
        training_data=training_data,
        epochs=1,
        mini_batch_size=10,
        learning_rate=0.5,
        regularization_parameter=0.1,
        evaluation_data=test_data, )
    results = [training_accuracy[0], evaluation_accuracy[0]]
    assert results == [10000, 1000]
