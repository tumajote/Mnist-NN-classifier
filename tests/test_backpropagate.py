import numpy as np
import pytest

from mnistclassifier.backpropagate import backpropagate
from mnistclassifier.network import Network


@pytest.fixture
def network():
    """Returns a Network instance with 3 layers of neurons: first 784, second 30 and third 10
    neurons in each"""
    return Network([784, 30, 10])


@pytest.fixture
def data():
    data = (np.zeros((784, 1)), np.zeros((10, 1)))
    return data


def test_backpropagate_results_are_in_correct_format(network, data):
    x, y = data
    nabla_bias, nabla_weights = backpropagate(network, x, y)
    shape = [nabla_bias[0].shape, nabla_bias[1].shape, nabla_weights[0].shape,
             nabla_weights[1].shape]

    assert shape == [(30, 1)
        , (10, 1)
        , (30, 784)
        , (10, 30)]
