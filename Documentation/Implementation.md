# Implementation

This architecture of the  mnist classifier is comprised of seven python modules: activation_functions, backpropagate, cost_functions, data, gradient_descent, network and plot. 

## Network

Instance of the Network class is the main data structure which holds all the relevant data for the neural net. The class constructor takes as argument a list of integers which depicts the architecture of the network. The first element of the list depicts the input layer and the last the output layer, the number elements between dictates the number of hidden layers. Each element is an integer which depicts the number of neurons in the layer. 

An instance of the Network object keeps track of the weights and biases of the network with numpy ndarrays which are instantiated during instantiation and have the dimensions dictated by the architecture of the network. The weights and biases are instantiated with random values as this gives better starting point for learning.

The actual classifying is done by the Network class’ method feedforward which takes an input: a picture of a digit as a vector of pixel densities and returns an output: a vector of ten elements of which each depicts a class of digits from 1 to 10. Each element in the output vector contains a value which depicts how confident the network is that the input is of that class. In other words, the method takes as an input a picture and outputs the networks guess of which digit the picture resembles.

The activation functions of the neurons are in their own module. 

## Training

The training of the network is done by the gradient_descent and backpropagate modules. 

The size of the mini batches, learning rate and regularization parameter are given as parameters to the train_with_stochastic_gradient_descent method. The method is also given an evaluation data set which is used to test how the network performs outside the training data. 

The method train_with_stochastic_gradient_descent divides the training set into mini batches. The method computes the gradient for cross entropy cost function with the method backpropagate for a given mini batch and updates the network’s weights and biases with the L2 regularization function. The cost and backpropagation functions are in their own modules. The same is done with every mini batch of the training data set. The amount of times the network trains with the whole training dataset is given by the epoch paramater. After each epoch the program prints the results of the training: the amount of correct classifications and the cost. Both of these figures are given for the training data and the evaluation data. 


## Data

The data module provides methods for loading the mnist dataset from a zip file and converting the data into a format that helps to utilize it in the classifier.

## Plotting 

The plot module provides functions for plotting the results as graphs. The plotting function provides graphs for cost and accuracy.

