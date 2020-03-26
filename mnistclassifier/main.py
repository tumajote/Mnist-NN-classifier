from mnistclassifier.network import Network
from mnistclassifier.data import format_data
from mnistclassifier.training import train

if __name__ == "__main__":
    test = Network([784, 30, 10])
    training_data, validation_data, test_data = format_data()
    """print("Training dataset length:",len(training_data))
    print("Training dataset length of first:", len(training_data[0]))
    print("Training dataset length of first:", len(training_data[0][1]))

    shape = []
    shape.append(len(training_data))
    shape.append(len(training_data[0]))
    shape.append(len(training_data[0][1]))

    for x in shape:
        print(x)"""
    train(test, training_data, 30, 10, 3.0, test_data)



