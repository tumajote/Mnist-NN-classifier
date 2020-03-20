This week I tried to build most of the code needed to build and train a neural network with the mnist dataset. I followed
mostly the example in the book "Neural Networks and Deep Learning" by Michael A. Nielsens. This took some time as the 
example code was in python2 and I wanted to write the code in python3. In some points I checked the requirements for
week two and realized I should focus also on setting up the project properly and make tests. I installed pytest for unit 
testing, coverage for test coverage reporting and flake8 for linting. I also spent many hours trying to setup tox for 
automating the build process, testing and linting but failed. It’s the first time I do any testing with python, so it 
took quite some time to figure out a proper setup and I did not have time left to do any actual tests. While working with 
the setup, I started to think about the structure of the project and realized that if I want to make some kind of application
out of the network, it does not maybe make sense to incorporate the training code into the application. So maybe the final 
application would include the trained network which could be used to recognize images? Or would it be better to also have 
the training phase in the application? I spent more than 20 hours working with this project this week. Next week I will focus
on unit testing and finalizing the training code.
