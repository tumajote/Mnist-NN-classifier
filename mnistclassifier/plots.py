import matplotlib.pyplot as plt
import numpy as np


def make_plots(evaluation_cost, evaluation_accuracy, training_cost,
               training_accuracy,
               num_epochs,
               training_cost_xmin=200,
               test_accuracy_xmin=200,
               test_cost_xmin=0,
               training_accuracy_xmin=0,
               training_set_size=50000,
               evaluation_set_size=10000
               ):
    plot_training_cost(training_cost, num_epochs, training_cost_xmin)
    plot_test_accuracy(evaluation_accuracy, num_epochs, test_accuracy_xmin,
                       evaluation_set_size)
    plot_test_cost(evaluation_cost, num_epochs, test_cost_xmin)
    plot_training_accuracy(training_accuracy, num_epochs,
                           training_accuracy_xmin, training_set_size)
    plot_overlay(evaluation_accuracy, training_accuracy, num_epochs,
                 min(test_accuracy_xmin, training_accuracy_xmin),
                 training_set_size, evaluation_set_size)


def plot_cost_general(cost, num_epochs, cost_xmin, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(cost_xmin, num_epochs),
            cost[cost_xmin:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title(title)
    plt.show()


def plot_training_cost(training_cost, num_epochs, training_cost_xmin):
    title = 'Cost on the training data'
    plot_cost_general(training_cost, num_epochs, training_cost_xmin, title)


def plot_test_cost(evaluation_cost, num_epochs, test_cost_xmin):
    title = 'Cost on the evaluation data'
    plot_cost_general(evaluation_cost, num_epochs, test_cost_xmin, title)


def plot_accuracy_general(accuracy, num_epochs, accuracy_xmin,
                          set_size, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(accuracy_xmin, num_epochs),
            [accuracy * 100.0 / set_size
             for accuracy in
             accuracy[accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title(title)
    plt.show()


def plot_test_accuracy(evaluation_accuracy, num_epochs,
                       evaluation_accuracy_xmin, evaluation_set_size):
    title = 'Accuracy (%) on the evaluation data'
    plot_accuracy_general(evaluation_accuracy, num_epochs,
                          evaluation_accuracy_xmin,
                          evaluation_set_size, title)


def plot_training_accuracy(training_accuracy, num_epochs,
                           training_accuracy_xmin, training_set_size):
    title = 'Accuracy (%) on the training data'
    plot_accuracy_general(training_accuracy, num_epochs,
                          training_accuracy_xmin, training_set_size, title)


def plot_overlay(evaluation_accuracy, training_accuracy, num_epochs, xmin,
                 training_set_size, evaluation_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_epochs),
            [accuracy * 100.0 / evaluation_set_size for accuracy in
             evaluation_accuracy],
            color='#2A6EA6',
            label="Accuracy on the evaluation data")
    ax.plot(np.arange(xmin, num_epochs),
            [accuracy * 100.0 / training_set_size
             for accuracy in training_accuracy],
            color='#FFA933',
            label="Accuracy on the training data")
    ax.grid(True)
    ax.set_xlim([xmin, num_epochs])
    ax.set_xlabel('Epoch')
    ax.set_ylim([90, 100])
    plt.legend(loc="lower right")
    plt.show()
