import numpy as np
import pytest

from mnistclassifier.data import format_data


@pytest.fixture
def data():
    training_data = ([[np.zeros(784)], [np.zeros(784)], [np.zeros(784)]],
                     [0, 0, 0])
    validation_data = ([[np.zeros(784)], [np.zeros(784)], [np.zeros(784)]],
                       [0, 0, 0])
    test_data = ([[np.zeros(784)], [np.zeros(784)], [np.zeros(784)]],
                 [0, 0, 0])
    data = (training_data, validation_data, test_data)

    return data


def test_data_is_in_formatted_correctly(data):
    training_data, validation_data, test_data = data
    training_data, validation_data, test_data = format_data(training_data,
                                                            validation_data,
                                                            test_data)
    shape = []
    shape.append(len(training_data))
    shape.append(len(training_data[0]))
    shape.append(len(training_data[0][0]))
    shape.append(len(training_data[0][1]))
    shape.append(len(validation_data))
    shape.append(len(validation_data[0]))
    shape.append(validation_data[0][1])
    shape.append(len(test_data))
    shape.append(len(test_data[0]))
    shape.append(test_data[0][1])

    assert shape == [3, 2, 784, 10, 3, 2, 0, 3, 2, 0]
