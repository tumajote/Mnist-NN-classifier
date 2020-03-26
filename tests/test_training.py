import numpy as np
import pytest

from mnistclassifier.network import Network


@pytest.fixture
def network():
    """Returns a Network instance with 3 layers of neurons: first 784, second 30 and third 10
    neurons in each"""
    return Network([784, 30, 10])