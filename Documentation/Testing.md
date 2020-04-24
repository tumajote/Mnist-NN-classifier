## First version

For the first version of the classifier (stochastic gradient descent, quadratic cost function, sigmoid activation function) I tried various hyper parameter settings. For example, with a network with one hidden layer of 30 neurons, minibatch size 10 and learning rate of 3.0 and training time of 30 epochs. The accuracy was around 95% as it classified 9484 images correctly. The training did not produce any significant increases in the accuracy after the 10th epoch. The network had input layer of 784 neurons, one hidden layer of 30 neurons and output layer of 10 neurons The training set consisted of 50000 images and the test set was 10000 images.

![Graph for the first version](https://github.com/tumajote/Mnist-NN-classifier/blob/master/Documentation/simple_network_results.png)

## Optimized version

In the optimized version I changed the way the weights are initialized, used cross entropy cost function and L2 regularization. The accuracy increased close to 98 %. The network had input layer of 784 neurons, one hidden layer of 100 neurons and output layer of 10 neurons. The hyperparameters were 50 epochs, minibatch size 10, learning rate 0.5 and the L2 regularization parameter 5.0. The network was trained with 50000 images and evaluated with 10000 images

![Graph for the optimised version 1]( https://github.com/tumajote/Mnist-NN-classifier/blob/master/Documentation/Optimazed_network_accuracy_evaluation_data.png)

I tried the same setup except learning rate was 0.2

![Graph for the optimised version 2]( https://github.com/tumajote/Mnist-NN-classifier/blob/master/Documentation/Optimazed_network_2_accuracy_evaluation_data.png)
