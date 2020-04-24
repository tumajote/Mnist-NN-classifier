import gzip
import pickle

import numpy as np

from mnistclassifier.gradient_descent import vectorized_result


def load_data_from_file():
    f = gzip.open('./mnist_dataset/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f,
                                                            encoding='latin1')
    f.close()
    return format_data(training_data, validation_data, test_data)


def format_data(training_data, validation_data, test_data):
    training_inputs = reshape_to_vector(training_data)
    training_results = [vectorized_result(y) for y in training_data[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = reshape_to_vector(validation_data)
    validation_results = validation_data[1]
    validation_data = zip(validation_inputs, validation_results)

    test_inputs = reshape_to_vector(test_data)
    test_results = test_data[1]
    test_data = zip(test_inputs, test_results)

    return list(training_data), list(validation_data), list(test_data)

def reshape_to_vector(data):
    return [np.reshape(x, (784, 1)) for x in data[0]]

