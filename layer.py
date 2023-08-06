import numpy as np
import function


class Layer:
    """
    Layer Object for neural network. Takes array of inputs or multiple array of inputs.
    This takes the form of a numpy matrix
    """

    def __init__(self, input_count, output_count):
        """
        Init method for layer object
        :param input_count: number of neurons feeding information into the layer
        :param output_count: number of output data points to next layer
        :type input_count, output_count: integer
        """
        self.input = None
        self.output = None
        self.weight_matrix = np.random.uniform(-0.5, 0.5, (output_count, input_count))

    def forward(self, input_matrix):
        """
        method to forward propagate data through the layer
        :param input_matrix: input matrix of outputs of neurons in the previous layer or init inputs.
                             For a single layer, the shape of this numpy matrix will be 1 x n
        :type input_matrix: numpy array of integers
        :return: numpy matrix of output values
        """
        self.input = np.asmatrix(input_matrix)
        self.output = np.dot(input_matrix, self.weight_matrix.T)
        self.output = function.soft_max(self.output)
        return self.output

    def backward_start(self, target):
        """
        backward propagation in layer object
        :param target: numpy array used to calculate new biases
        """
        if len(self.output) != len(target):
            raise IndexError("array lengths must be the same")
        diff = []
        for o, t in zip(self.output, target):
            diff.append(o - t)
        diff = np.asmatrix(diff)
        self.output = diff
        self.weight_matrix -= 0.01 * np.dot(diff.T,self.input)

    def backward_next(self,diff,weight_matrix):
        """
        backward propagation in the layer object
        :param diff: delta array of forward layer
        :param weight_matrix: matrix of weights for forward layer
        """
        temp = self.output * (1 - self.output)
        temp = np.asmatrix(temp)
        front = np.dot(diff,weight_matrix)
        temp = np.asarray(temp)
        front = np.asarray(front)
        delta = front * temp
        self.output = delta
        self.weight_matrix -= 0.01 * np.dot(delta.T, self.input)


