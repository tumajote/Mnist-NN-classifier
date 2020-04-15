import matplotlib.pyplot as plt

from mnistclassifier.data import format_data, load_data_from_file
from mnistclassifier.gradient_descent \
    import train_with_stochastic_gradient_descent
from mnistclassifier.network import Network

if __name__ == "__main__":
    test = Network([784, 30, 10])
    data = load_data_from_file()
    training_data, validation_data, test_data = format_data(data)
    number_of_correct_results, percent_of_correct_results \
        = train_with_stochastic_gradient_descent(
        test, training_data, 30,
        10, 3.0, test_data)
    plt.plot(number_of_correct_results)
    plt.ylabel("Number of correct classifications")
    plt.xlabel("Epoch")
    plt.ylim(top=10000)
    plt.show()

    plt.plot(percent_of_correct_results)
    plt.ylabel("Percent of correct classifications")
    plt.xlabel("Epoch")
    plt.ylim(top=100)
    plt.show()
