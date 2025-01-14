from data import get_mnist # import function to get the MNIST dataset
import numpy as np # for numerical computations
import matplotlib.pyplot as plt # for visualization of images and results

"""
w = weights, i = input layer, o = output layer, b = bias, h = hidden layer, l = label

w_i_h = weights from input layer to hidden layer
w_h_o = weights from hidden layer to output layer
"""

# load the MNIST dataset containing images and labels
images, labels = get_mnist()

# initialize weights(random values between -0.5 and 0.5) and biases(zeros)
# describe the neural network
# input layer to hidden layer : 20 hidden neurons each connected to the 784 input neurons(28x28)
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))

# hidden layer to output layer : 10 output neurons each connected to the 20 hidden neurons
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))

# bias for hidden layer : 20 hidden neurons initialized to 0
b_i_h = np.zeros((20, 1))

# bias for output layer : 10 output neurons intialized to 0
b_h_o = np.zeros((10, 1))

# set learning rate(step size for weight updates)
learn_rate = 0.01

# initialize counter to track the number of correct predictions
nr_correct = 0

# set how many times to go through the entire dataset
epochs = 3

# loop through the dataset for the specified number of epochs
for epoch in range(epochs):
    for img, l in zip(images, labels): # loop throught each image and its label
        # reshape the image and label to column vectors
        img.shape += (1,) # add dimension to image(784x1)
        l.shape += (1,) # add dimension to lable(10x1) one-hot encoded

        # forward propagation: input layer -> hidden layer
        h_pre = b_i_h + w_i_h @ img # calculate the weighted sum for the hidden layer
        h = 1 / (1 + np.exp(-h_pre)) # apply sigmoid activation to introduce non-linearity

        # forward propagation: hidden layer -> output layer
        o_pre = b_h_o + w_h_o @ h # calculate wieghted sum for the output layer
        o = 1 / (1 + np.exp(-o_pre)) # apply sigmoid activation

        # cost/error calculation: Mean squared error
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0) # average squared difference between predicted and actual values

        # check if the predicted output matches the label(using argmax to get the index of the largest value)
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # backpropagation: output layer -> hidden layer
        delta_o = o - l # compute gradient error with respect to the output layer
        w_h_o += -learn_rate * delta_o @ np.transpose(h) # update weights(gradient decent)
        b_h_o += -learn_rate * delta_o # update biases for the output layer

        # backpropagation: hidden layer -> input layer
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h)) # compute the gradient for the hidden layer
        w_i_h += -learn_rate * delta_h @ np.transpose(img) # update weights for the hidden layer
        b_i_h += -learn_rate * delta_h # update biases for the hidden layer

    # display the accuracy after each epoch
    print(f"Accuracy: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0 # reset counter for the next epoch



