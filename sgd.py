import random


def SGD(network, training_data, epochs, mini_batch_size, eta, test_data=None):
    """Training the network with mini-batch stochastic gradient descent"""
    if test_data: n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k + mini_batch_size]
            for k in range(0, n, mini_batch_size)
        ]
        for mini_batch in mini_batches:
            update_mini_batch(mini_batch, eta)
        if test_data:
            print
            "Epoch {0}: {1} / {2}".format(j, evaluate(test_data), n_test)
        else:
            print
            "Epoch {0} complete".format(j)


def update_mini_batch(network, eta):
    return network


def evaluate(test_data):
    return True
