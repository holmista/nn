# MNIST Classification from scratch

## Motivation

This project is all about classifying the MNIST dataset by building a machine learning model from scratch
without using any machine learning library (however I did use numpy).
For starters, the MNIST dataset is a dataset of 60,000 handwritten 28x28 grayscale images of the 10 digits, along with a test set
of 10,000 images. It is a widely used dataset for training and testing machine learning models.

Usually MNIST is used as one of the introductory dataset for beginners when learning computer vision and machine learning, however I
wanted to take it a step further and actually see how the black box magic of machine learning libraries work under the hood. 

## About the model itself

In theory the model could work on many other datasets other than MNIST (you're free to experiment on those!), since I tried to make it as general
and flexible as possible in the scope of this project, however I have only tested it on 1 other dataset other than MNIST so I can't tell for sure.

Okay, back to the MNIST classification, the model architecture is quite simple, it is a MLP (Multi Layer Perceptron) with 4 layers, 1 input layer,
2 hidden layers and 1 output layer. The hidden layers use ReLU activation and the output layer uses softmax. All layers are fully connected.
Now, instead of using something like convolutional neural networks (CNNs) which are more suited for image classification tasks, I wanted to keep it
simple and used a simple MLP which was not a bad decision at all since the model achieved up to 90% accuracy on the test set which is quite good
for a simple model like this.

## Training the model

For training I used a standard forward and backward propagation algorithm with mini-batch gradient descent. The hidden layers use the ReLU activation
and generally they were easy to handle for both forward and backward propagation. The output layer caused me some trouble since I used softmax.
It's really easy to implement softmax in theory but in my case because of a small bug in the implementation it caused exploding gradients problem which
was causing numerical instability in the whole model. Only after 2 days of debugging I found the bug and fixed it. The model was then able to train properly.  

## About using dependencies

In the beginning I wanted to use 0 external dependencies and build everything from scratch, and initially I did that, I did not use numpy and implemented
all the linear algebra operations I needed in good old for loops, this worked for the little toy dataset I used as a sanity check but when I tried to train
the model on the actual MNIST dataset it was taking too long to train, so I decided to use numpy for the linear algebra operations. And since it's possible to
build other models from this project I decided it was worth it to use numpy.