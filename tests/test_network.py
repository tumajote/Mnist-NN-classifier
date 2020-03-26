import numpy as np
import pytest

from mnistclassifier.network import Network


@pytest.fixture
def network():
    """Returns a Network instance with 3 layers of neurons: first 784, second 30 and third 10
    neurons in each"""
    return Network([784, 30, 10])


def test_network_has_correct_number_of_layers(network):
    assert network.num_layers == 3


def test_network_has_correct_number_of_biases(network):
    assert sum([len(x) for x in network.biases]) == 40


def test_network_has_correct_number_of_weights(network):
    count = 0
    for x in network.weights:
        for y in x:
            for z in y:
                count += 1
    assert count == 23820


def test_network_feedforward_method_produces_correct_number_of_outputs(network):
    assert len(network.feedforward(np.zeros((784, 1)))) == 10
