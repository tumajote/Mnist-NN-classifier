# Performance testing 

## First version

For the first version of the classifier (stochastic gradient descent, quadratic cost function, sigmoid activation function) I tried various hyper parameter settings. For example, with a network with one hidden layer of 30 neurons, minibatch size 10 and learning rate of 3.0 and training time of 30 epochs. The accuracy was around 95% as it classified 9484 images correctly. The training did not produce any significant increases in the accuracy after the 10th epoch. The training set consisted of 50000 images and the test set was 10000 images.

![Graph for the first version](https://github.com/tumajote/Mnist-NN-classifier/blob/master/Documentation/simple_network_results.png)
