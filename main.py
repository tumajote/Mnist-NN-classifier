import argparse

from mnistclassifier.cost_functions import CrossEntropyCost
from mnistclassifier.data import format_data, load_data_from_file
from mnistclassifier.gradient_descent \
    import train_with_stochastic_gradient_descent
from mnistclassifier.network import Network
from mnistclassifier.plots import make_plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a simple Mnist'
                                                 'classifier')
    parser.add_argument('--epochs', type=int, default=10,
                        help='How many epochs to train')
    parser.add_argument('--mini_batch_size', type=int, default=10,
                        help='How many training samples in a mini batch')
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='Set the learning rate')
    parser.add_argument('--regularization_parameter', type=float, default=5.0,
                        help='Sets the L2 regularization parameter')
    args = parser.parse_args()

    test = Network([784, 100, 10], cost=CrossEntropyCost)
    data = load_data_from_file()
    training_data, validation_data, test_data = format_data(data)

    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy = train_with_stochastic_gradient_descent(
        network=test,
        training_data=training_data,
        epochs=args.epochs,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.learning_rate,
        regularization_parameter=args.regularization_parameter,
        evaluation_data=test_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)

    if args.epochs > 1:
        make_plots(evaluation_cost, evaluation_accuracy, training_cost,
                   training_accuracy,
                   num_epochs=args.epochs,
                   training_cost_xmin=0,
                   test_accuracy_xmin=0,
                   test_cost_xmin=0,
                   training_accuracy_xmin=0,
                   training_set_size=50000)
