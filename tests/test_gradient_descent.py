import numpy as np
import pytest


@pytest.fixture
def network():
    """Returns a Network instance with 3 layers of neurons: first 784, second 30 and third 10
    neurons in each"""
    return Network([784, 30, 10])


@pytest.fixture
def data():
    """Training data"""
    n = 100
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

    """Test data"""
    n = 1000
    x = [np.zeros((784, 1)) for x in range(n)]
    y = [0 for x in range(n)]
    test_data = list(zip(x, y))
    for x, y in test_data:
        for i in range(0, 200):
            x[i] = 0.99

    x = [np.zeros((784, 1)) for x in range(n)]
    y = [3 for x in range(n)]
    test_data_extra = list(zip(x, y))
    for x, y in test_data_extra:
        for i in range(300, 500):
            x[i] = 0.99
    test_data.extend(test_data_extra)

    data = training_data, test_data

    return data
