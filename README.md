# neural_network_imaging_experiments
This repository contains and details some experiments I have done. 

The basic idea of these projects are to train a neural network to output a certain image based on some input format and then seeing what outputs we get if we ask the nerual net to produce the image for an input value not in it's training set. 

Further I wanted to implement backpropagation learning and a neural network. Specifically this network uses the stochastic gradient desent learning algorithm with no biases with an sigmoid activation funciton with one global learning rate which adaptively scales down based on a simple algorithm (half it if the average error the past 100 epochs are greater than the previous 100 epochs). I plan to try a couple of different things, if I do I will document them here. 

This repository is not intended to be used as a library for other applications (although if you'd like you're free to use it however you'd like) nor as research but rather to document a process of artistic expression through the generation of images by a neural network. 

The training input folder shows the images the network was trained on. The file name is the weights to the eight input nodes. This input format was used in version 0.1.6 and 0.1.6.1. Each input can be thought of as a button that is pressed. 

0.1.6 and 0.1.6.1 contains the outputs of a neural net with two hidden layers, with the following sizes: 8, 110, 110, 110 times 110 where 110x110 is the output format. 
0.1.6 does not have adaptive scaling of learning rate, 0.1.6.1 has. Both are trained for 10000 epochs on the eight input images. Both have a learning parameter of (initially) 0.1

The files with "filename" in them are the outputs given by the neural network during the last training epoch (meaning that the weights may have changed slightly before the rest of the output). If we compare 0.1.6 and 0.1.6.1 we can see that the neural network is better at different images. This is probably because the adapting learning rate version of the neural network got stuck on a hypothesis favoring these images, while the constant learning rate managed to move to another hypothesis favoring some other images.

The "sample" folders show some of the output images that I liked the most. 

If we consider what function we are trying to approximate we see that we are essentially trying to make the neural network learn a look-up table of pixel values. Indeed, if we change the network structure to having merely two layers, one with the 8 input nodes and one with 110x110 output nodes the neural network quickly converges to far lower error values than with the tested deeper network structure. Interestingly, the output images in this case are also interesting visually. This is not due to some interesting "thinking" in the neural network, but rather that a sigmoid function called on a sum of pixel values can produce interesting outputs. 
