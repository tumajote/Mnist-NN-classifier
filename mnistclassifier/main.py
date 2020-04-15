from mnistclassifier.data import format_data, load_data_from_file
from mnistclassifier.gradient_descent \
    import train_with_stochastic_gradient_descent
from mnistclassifier.network import Network

if __name__ == "__main__":
    test = Network([784, 30, 10])
    data = load_data_from_file()
    training_data, validation_data, test_data = format_data(data)
    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy = train_with_stochastic_gradient_descent(
        network=test,
        training_data=training_data,
        epochs=30,
        mini_batch_size=10,
        learning_rate=3.0,
        regularization_parameter=0.1,
        evaluation_data=test_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
