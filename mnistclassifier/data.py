import gzip
import pickle

import numpy as np

from mnistclassifier.gradient_descent import vectorized_result


def load_data_from_file():
    """Load the MNIST dataset from a zip file"""
    f = gzip.open('./mnist_dataset/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f,
                                                            encoding='latin1')
    f.close()
    return format_data(training_data, validation_data, test_data)


def format_data(training_data, validation_data, test_data):
    """Formats the dataset to a suitable format. The image matrices
     are formatted into vectors with the pixel densities as the elements. This
      format matches the input layer of the network. For the training date set
      the output is also vectorized to match the networks output layer which
      enables the computing of the cost function"""
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
    """Reshape the image matrix into a vector"""
    return [np.reshape(x, (784, 1)) for x in data[0]]
