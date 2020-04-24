import matplotlib.pyplot as plt
import numpy as np


def make_plots(evaluation_cost, evaluation_accuracy, training_cost,
               training_accuracy,
               num_epochs,
               training_set_size=50000,
               evaluation_set_size=10000
               ):
    """Draw plots for the cost and accuracy. The first epoch is numbered 0"""

    """Adjust the epochs shown in the plot. The variable gives the starting
    epoch"""
    starting_epoch = 0

    """Draw plots if there is more than one epoch"""
    if num_epochs > 1:
        """Cost over epochs for training data"""
        plot(training_cost, num_epochs, starting_epoch,
             title='Cost on the training data')

        """Cost over epochs for evaluation data"""
        plot(evaluation_cost, num_epochs, starting_epoch,
             title='Cost on the evaluation data')

        """Accuracy over epochs for training data"""
        plot(training_accuracy, num_epochs, starting_epoch,
             'Accuracy (%) on the training data', training_set_size)

        """Accuracy over epochs for evaluation data"""
        plot(evaluation_accuracy, num_epochs, starting_epoch,
             'Accuracy (%) on the evaluation data',
             evaluation_set_size)

        """Accuracy over epochs for evaluation and training data overlaid"""
        plot_overlay(evaluation_accuracy, training_accuracy, num_epochs,
                     starting_epoch, training_set_size, evaluation_set_size)

        """Cost over epochs for evaluation and training data overlaid"""
        plot_overlay(evaluation_cost, training_cost, num_epochs,
                     starting_epoch)


def plot(data_array, num_epochs, starting_epoch, title, data_set_size=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if data_set_size:
        ax.plot(np.arange(starting_epoch, num_epochs),
                [accuracy * 100.0 / data_set_size
                 for accuracy in
                 data_array[starting_epoch:num_epochs]],
                color='#2A6EA6')
    else:
        ax.plot(np.arange(starting_epoch, num_epochs),
                data_array[starting_epoch:num_epochs],
                color='#2A6EA6')
    ax.set_xlim([starting_epoch, num_epochs - 1])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title(title)
    if num_epochs < 6:
        make_ticks_integers()
    plt.show()


def plot_overlay(data_array_1, data_array_2, num_epochs,
                 starting_epoch,
                 training_set_size=None, evaluation_set_size=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if training_set_size and evaluation_set_size:
        ax.plot(np.arange(starting_epoch, num_epochs),
                [accuracy * 100.0 / evaluation_set_size for accuracy in
                 data_array_1],
                color='#2A6EA6',
                label='Accuracy on the evaluation data')
        ax.plot(np.arange(starting_epoch, num_epochs),
                [accuracy * 100.0 / training_set_size
                 for accuracy in data_array_2],
                color='#FFA933',
                label='Accuracy on the training data')
        ax.set_ylim([50, 100])
    else:
        ax.plot(np.arange(starting_epoch, num_epochs),
                data_array_1[starting_epoch:num_epochs],
                color='#2A6EA6',
                label='Cost on the evaluation data')
        ax.plot(np.arange(starting_epoch, num_epochs),
                data_array_2[starting_epoch:num_epochs],
                color='#FFA933',
                label='Cost on the training data')
    ax.grid(True)
    ax.set_xlim([starting_epoch, num_epochs - 1])
    ax.set_xlabel('Epoch')
    plt.legend(loc="lower right")
    if num_epochs < 6:
        make_ticks_integers()
    plt.show()


def make_ticks_integers():
    xint = []
    locs, labels = plt.xticks()
    for each in locs:
        xint.append(int(each))
    plt.xticks(xint)
