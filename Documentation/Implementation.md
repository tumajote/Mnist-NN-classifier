# Implementation

This architecture of the  mnist classifier neural net is comprised of five python modules: activation functions, backpropagate, data, gradient_descent and network. 

## Network

Instance of the Network class is the main data structure which holds all the relevant data for the neural net. The class constructor takes as argument a list of integers which depicts the architecture of the network. The first element of the list depicts the input layer and the last the output layer, the number elements between dictates the number of hidden layers. Each element is an integer which depicts the number of neurons in the layer. 

An instance of the Network object keeps track of the weights and biases of the network with numpy ndarrays which are instantiated during instantiation and have the dimensions dictated by the architecture of the network. The weights and biases are instantiated with random values as this gives better starting point for learning.

The actual classifying is done by the Network class’ method feedforward which takes an input: a picture of a digit as a vector of pixel densities and returns an output: a vector of ten elements of which each depicts a class of digits from 1 to 10. Each element in the output vector contains a value which depicts how confident the network is that the input is of that class. In other words, the method takes as an input a picture and outputs the networks guess of which digit the picture resembles.

The activation functions of the neurons are in their own module. 

## Training

The training of the network is done by the gradient_descent and backpropagate modules. 

The method train_with_stochastic_gradient_descent divides the training set into minibatches which size can be given as parameter. The method computes the gradient for quadratic cost function with the method backpropagate for a given minibatch and updates the weights and biases accordingly. The learning rate can also be adjusted by giving a parameter to the train_with_stochastic_gradient_descent method. 


## Data

The data module provides methods for loading the mnist dataset from a zip file and converting the data into an format that helps to utilize it in the classifier.

