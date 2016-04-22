Digit Recognition using Artificial Neural Networks

== Objective ==

The objective of this problem is to perform digit recognition given an image of a
handwritten digit using Artificial Neural Networks (ANN). This problem has 2 parts:
Perform the digit recognition using a baseline neural network and then modify the
network by tweaking its hyper parameters and report the results in each of the
cases.
You are provided with a subset of the MNIST dataset (which is a large database of
handwritten digits commonly used for training various image processing systems)
for this task.

The broad steps to be followed are as follows:

* normalize the images to a standard size
* Implement the ANN
* Build and train the network.
* Test the baseline network developed above and report the accuracy
* Tune the network by tweaking hyper parameters using the validation data,
  test and report accuracies for each of the cases.
  
== Procedure followed ==

We normalized the image using the PIL library in python and resized all the images to
16x8 and converted the image into a vector containing RGB values of length 384.We
split the dataset into 70% for training, 15% for validation and 15% for testing as
suggested and ran the network with 4 and 8 hidden units in the hidden layer.We varied the
number of epochs along with the learning rate during training and observed the
results that we obtained

== Inferences made ==

The effect of a lower learning rate does not affect the accuracy highly when the
number of epochsare increased. In fact, having a higher learning rate gave a lower
accuracy when the number of epochs were high. This was probably because the right
values of the weight matrices could not be reached due to the precision being low.
The Neural Network with a higher number of hidden units also needed more epochs
to train than ones with lower hidden units but yielded a relatively higher accuracy
value of 23%.
