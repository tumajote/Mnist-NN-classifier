from mnistclassifier import load_data
from mnistclassifier.network import Network
from mnistclassifier.training import train

if __name__ == "__main__":
    test = Network([784, 30, 10])
    training_data, validation_data, test_data = load_data()
    train(test, training_data, 30, 10, 3.0, test_data)
