from mnistclassifier.data import format_data, load_data_from_file
from mnistclassifier.network import Network

if __name__ == "__main__":
    test = Network([784, 30, 10])
    data = load_data_from_file()
    training_data, validation_data, test_data = format_data(data)
    x, y = training_data[0]
    print(x)
    print(y)
#    bp = backpropagate(test, x, y)
    print(bp)

    # train(test, training_data, 30, 10, 3.0, test_data)
