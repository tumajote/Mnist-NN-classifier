import pytest
from mnistclassifier.data import format_data


@pytest.fixture
def data():
    """Returns a Network instance with 3 layers of neurons: first 784, second 30 and third 10
    neurons in each"""
    return format_data()


def test_training_data_is_in_correct_format(data):
    training_data, validation_data, test_data = data
    shape = []
    shape.append(len(training_data))
    shape.append(len(training_data[0]))
    shape.append(len(training_data[0][1]))

    assert shape == [50000, 2, 10]


