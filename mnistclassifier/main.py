from mnistclassifier.cost_functions import QuadraticCost,CrossEntropyCost
from mnistclassifier.data import format_data, load_data_from_file
from mnistclassifier.gradient_descent \
    import train_with_stochastic_gradient_descent
from mnistclassifier.network import Network
from mnistclassifier.plots import make_plots


if __name__ == "__main__":
    test = Network([784, 100, 10], cost=CrossEntropyCost)
    data = load_data_from_file()
    training_data, validation_data, test_data = format_data(data)
    epochs= 1
    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy = train_with_stochastic_gradient_descent(
        network=test,
        training_data=training_data,
        epochs=epochs,
        mini_batch_size=10,
        learning_rate=0.5,
        regularization_parameter=5.0,
        evaluation_data=test_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
    make_plots(evaluation_cost, evaluation_accuracy, training_cost,
               training_accuracy,
               num_epochs=epochs,
               training_cost_xmin=0,
               test_accuracy_xmin=0,
               test_cost_xmin=0,
               training_accuracy_xmin=0,
               training_set_size=50000)