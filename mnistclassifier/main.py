import numpy as np

from mnistclassifier.data import format_data, load_data_from_file
from mnistclassifier.network import Network
from mnistclassifier.gradient_descent import train_with_stochastic_gradient_descent

if __name__ == "__main__":
    test = Network([784, 30, 10])
    data = load_data_from_file()
    training_data, validation_data, test_data = format_data(data)
    train_with_stochastic_gradient_descent(test, training_data, 3, 10, 3.0, test_data)