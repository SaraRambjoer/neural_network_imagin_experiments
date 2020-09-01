# neural_network_imaging_experiments
This repository contains and details some experiments I have done. 

The basic idea of these projects are to train a neural network to output a certain image based on some input format and then seeing what outputs we get if we ask the nerual net to produce the image for an input value not in it's training set. 

Further I wanted to implement backpropagation learning and a neural network. Specifically this network uses the stochastic gradient desent learning algorithm with no biases with an sigmoid activation funciton with one global learning rate which adaptively scales down based on a simple algorithm (half it if the average error the past 100 epochs are greater than the previous 100 epochs). I plan to try a couple of different things, if I do I will document them here. 

This repository is not intended to be used as a library for other applications (although if you'd like you're free to use it however you'd like) nor as research but rather to document a process of artistic expression through the generation of images by a neural network. 
