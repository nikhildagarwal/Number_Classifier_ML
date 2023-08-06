import numpy as np


def get_label(num):
    """
    creates label matrix to calculate and loss and begin back propagation step
    :param num: the number that we are actually trying to identify with the neural network. Given to us via test cases.
    :return: label matrix of size 1 x 10 where every index is 0 except for the index of the correct number
    """
    if num > 9:
        raise IndexError("Only identifies single digits between 0 and 9 inclusive!")
    label_matrix = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    label_matrix[num] = 1
    return label_matrix
