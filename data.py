import numpy as np
import pathlib


def get_training_data():
    """
    extracts training data from npz file
    :return: array of image pixel data (grayscale) and array of labels for training purposes
    """
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as fi:
        images, labels = fi["x_train"], fi["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels


def get_test_data():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as fi:
        images, labels = fi["x_test"], fi["y_test"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels

