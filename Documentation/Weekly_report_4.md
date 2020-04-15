This week I managed to get the test coverage to near 100% for the relevant code. This required the construction of mock training set and test set and then to test if the training algorithms could learn from these sets. I build a training set which included only two distinct images (of blocks). I trained the network with this dataset to see if the training would enable the network to classify these two images correctly. This experiment was small enough to run as a software test and enabled me to test the gradient descent and backpropagation algorithms as I could not figure out how to test them otherwise. This took quite a while and most of my hours were spent on this. Afterwards I wrote a sketch of the implementation document and started to plan how to do performance testing for the project. As the performance of neural net is quite a complex matter, I’m not sure what to measure. The obvious thing to measure is the amount of correct classifications the network produces but this metric depends on the hyperparameters. Consequently, there is no way to compare thoroughly the performance of the network. As I did not know what to measure I just posted an example metric of a setup I tried with my network. Next week I’m not sure what should I start working on? The basic setup is now working and tested. I’m thinking of starting to optimize the network for example by implementing different activation and cost functions and also implementing better metrics for the network to detect overfitting.I spent 16 hours this week.