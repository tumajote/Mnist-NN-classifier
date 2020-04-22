I spent a lot of time to do the code review as I find doing them a great way to learn. After that I implemented few optimizations from Nielsen’s book to the network. I rebuild the network class to enable the changes and added some plotting functions to visualize the results. I changed the way the weights are initialized so that the network begins to learn more quickly. I added another cost function called CrossEntropyCost. I also added a L2 regularization which prevents overfitting. Then I spend quite many hours trying to understand what these optimizations actually do which was quite difficult because of the heavy math involved. I ran the optimized network and put the results in the test documentation.Then I refactored the unittests to suit the new network. In the end I refactored the main class to enable the program to be run from terminal with given parameters.I also wrote a short user guide on how to run the training from terminal. 

I’m quite sure that there is not much optimizing to be done anymore so I would need some advice what to do next. The main functionality is based on numpy arrays, which I cannot replace without making the network obsolete. The network uses in some places python’s standard list data structure and zip method. Should I implement the list data structure from scratch and if yes can I use tuples or arrays? This could (not sure) also make the network not work efficiently enough as the list is mainly used in the backpropagate function which involves quite heavy computing. I could also just replace the lists with tuples or arrays with little effort (because I guess I will always know their length in advance) but that seems a little redundant.  Another data structure I could try to build from scratch is the zip method and the object it produces which is utilized quite heavily in the code, but I guess that would hamper the performance also quite heavily.