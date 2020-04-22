## Train the network (requires Python 3)
Pull the repo navigate, to the root and install dependencies (if you don't want to install dependencies globally
initialize a virtual environment like venv) 
```
Pip install -r requirements.txt
```
Train the network with default settings (10 epochs, minibatch size is 10, learning rate is 0.5 and the L2 regularization
parameter is 5.0)
```
python3 main.py
```
If you want to tweak the hyperparameters you can. Epochs and mini batch size should be integers and learning rate and
regularization parameter floats. For example: 
```
python3 main.py --epochs=20 --mini_batch_size=20 --learning_rate=3.0 --regularization_parameter=6.0
```
The arguments are optional so you can initiate them in any combination.

### Output
The program will output the amount of correct classifications per epoch. If there is more than one epoch of training,
athe program will plot four different graphs that visualize the accuracy and cost on the training and evaluation data.

### Network architecture 
Currently the network has input layer of 784 neurons, one hidden layer of 100 neurons and output layer of 10 neurons.
The training is done with stochastic gradient descent, backpropagation and cross entropy cost function. In the future these
might become customizable from the terminal.

### Training and evaluation data
The network is trained with 50000 images and evaluated with 10000 images both from MNIST dataset.
