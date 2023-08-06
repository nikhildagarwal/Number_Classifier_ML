import math
import numpy as np


def soft_max(inputs):
    """
    sigmoid function to keep all outputs values between 0 and 1
    :param inputs: numpy array of integers - size 1 x n
    :return numpy array of integers - size 1 x n, after adjusting weights
    """
    output = []
    for num in inputs:
        outNum = 1 / (1 + math.e ** (-1 * num))
        output.append(outNum)
    return np.array(output)


def normalize(inputs):
    """
    Normalize the output of the final layer.
    Makes it so that the sum of the outputs adds up to 1
    :param inputs: numpy array of integers - size 1 x n
    :return: numpy array of integers - size 1 x n, after normalizing
    """
    output = []
    base = sum(inputs)
    for num in inputs:
        output.append(num / base)
    return output


def mean_squared_error(outputs, target):
    """
    function to calculate the mean squared error of our predicted array value to the target values.
    :param outputs: np array of normalized values from neural network
    :param target: np array of target values with correct index marked with a 1
    :return:
    """
    if len(outputs) != len(target):
        raise IndexError("arrays must be the same length")
    coeff = 1 / len(outputs)
    outSum = 0
    for fi, yi in zip(target, outputs):
        outSum += (fi - yi)**2
    return coeff * outSum
