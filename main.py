from data import load_data
from network import Network
from training import train

if __name__ == "__main__":
    test = Network([784, 30, 10])
    training_data, validation_data, test_data = load_data()
    train(test, training_data, 30, 10, 3.0, test_data)
