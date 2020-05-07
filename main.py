import argparse

from mnistclassifier.cost_functions import CrossEntropyCost
from mnistclassifier.data import load_data_from_file
from mnistclassifier.gradient_descent \
    import train_with_stochastic_gradient_descent
from mnistclassifier.network import Network
from mnistclassifier.plots import make_plots

if __name__ == "__main__":
    """Set the command line parsing and the parameters to train the network"""
    parser = argparse.ArgumentParser(description='Train a simple Mnist'
                                                 'classifier')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='How many epochs to train')
    parser.add_argument('-m', '--mini_batch_size', type=int, default=100,
                        help='How many training samples in a mini batch')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.5,
                        help='Set the learning rate')
    parser.add_argument('-r', '--regularization_parameter', type=float,
                        default=5.0,
                        help='Sets the L2 regularization parameter')
    parser.add_argument('-i', '--hidden_layers', nargs='+', type=int,
                        default=[100],
                        help='Set the amount of hidden'
                             ' layers and their neurons')
    args = parser.parse_args()

    """Set the input and output layers to match the data"""
    layers = [784]
    layers.extend(args.hidden_layers)
    layers.append(10)

    """Initiate the network"""
    test = Network(layers, cost=CrossEntropyCost)
    training_data, validation_data, test_data = load_data_from_file()

    """Train the network"""
    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy = train_with_stochastic_gradient_descent(
        network=test,
        training_data=training_data,
        epochs=args.epochs,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.learning_rate,
        regularization_parameter=args.regularization_parameter,
        evaluation_data=test_data)

    """Draw graphs if there is more than one epoch"""
    if args.epochs > 1:
        make_plots(evaluation_cost, evaluation_accuracy, training_cost,
                   training_accuracy,
                   num_epochs=args.epochs,
                   training_set_size=50000)
