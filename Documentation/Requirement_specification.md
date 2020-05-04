# Requirement specification
A simple program for creating and training feedforward multilayer neural networks for classifying images in the Mnist dataset. The user will be able to choose the amount of hidden layers and the amount of neurons in the layers. The user can set various hyperparameters for the training phase such as mini batch size, amount of epochs, learning rate and L2 regularization parameter. After each epoch the program presents the accuracy and cost for the training dataset and also for a separate evaluation dataset. After the training is finished the program presents plots for the cost and accuracy by epochs and datasets. 

### Main algorithms

- For classification the network uses feedforward propagation which uses a sigmoid function as the activation function
- For training the network, the program uses cross entropy cost function with weight decay or L2 regularization. During training the weights and biases are updated with the stochastic gradient descent and the gradient is computed with backpropagation

I use [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen Determination Press, 2015 as a guide in my work.

### Time complexity

For using the trained network to do classification the time complexity is derived from the feedforward propagation, which has time complexity of O(n^4).

For training the network the time complexity derives from the backpropagation algorithm and has the time complexity of O(n^5).

[Source]( https://kasperfred.com/series/introduction-to-neural-networks/computational-complexity-of-neural-networks)
