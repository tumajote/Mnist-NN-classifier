from mnistclassifier.cost_functions import QuadraticCost,CrossEntropyCost
from mnistclassifier.data import format_data, load_data_from_file
from mnistclassifier.gradient_descent \
    import train_with_stochastic_gradient_descent
from mnistclassifier.network import Network

if __name__ == "__main__":
    test = Network([784, 100, 10], cost=CrossEntropyCost)
    data = load_data_from_file()
    training_data, validation_data, test_data = format_data(data)
    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy = train_with_stochastic_gradient_descent(
        network=test,
        training_data=training_data,
        epochs=10,
        mini_batch_size=10,
        learning_rate=0.5,
        regularization_parameter=0.1,
        evaluation_data=test_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
