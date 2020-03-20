from src.mnistclassifier.data import load_data
from src.mnistclassifier.network import Network
from src.mnistclassifier.training import train

if __name__ == "__main__":
    test = Network([784, 30, 10])
    training_data, validation_data, test_data = load_data()
    train(test, training_data, 30, 10, 3.0, test_data)
