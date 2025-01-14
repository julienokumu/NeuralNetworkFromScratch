# Import numpy for mathematical computations. In this case for matrix manipulations, data normalization and one-hot encoding
import numpy as np
# Import pathlib for file path manipulations
import pathlib

# Function that loads the MNIST dataset stored as a .npz file
def get_mnist():
    # np.load function loads the MNIST dataset
    # pathlib is dynamically constructed for file pathing
    # __file__ refers to the current file's path
    # .parent.absolute() gets the absolute path to the folder containing this script
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        # Extract training images, x_train and labels, y_train from the loaded file
        images, labels = f["x_train"], f["y_train"]

    # Normalizie the pixel values of the images to the range[0,1] by dividing by 255(max pixel value) 
    # To ensure neural network learns faster and more effectively since inputs are on a smaller scale
    images = images.astype("float32") / 255
    # Reshape each 28x28 image into a flat array of 784 pixels
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    # Convert labels inot one-hot encooded vectores
    # e.g label 3 is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # np.eye(10) creates a 10x10 identity matrix where each row corresponds to a one-hot vector
    labels = np.eye(10)[labels]

    # Return preproccessed images and one-hot encoded labels as a tuple
    return images, labels