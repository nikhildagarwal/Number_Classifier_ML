import layer
from data import get_mnist
import numpy as np

print("Running...")
images, labels = get_mnist()

"""
Initialize Neural layers.
Neural network consists of 2 hidden layers of 392 neurons and 100 neurons each.
"""
l1 = layer.Layer(784,392)
l2 = layer.Layer(392,100)
l3 = layer.Layer(100,10)

"""
Loops 3 times over the entire dataset of 60000 pictures.
Prints the overall accuracy of our network after each iteration
"""
for i in range(3):
    number_of_correct = 0
    total_number = 60000
    for img, label in zip(images, labels):

        """
        Forward propagation
        """
        l1.forward(img)
        l2.forward(l1.output)
        l3.forward(l2.output)

        pred_max = np.argmax(l3.output)
        target_max = np.argmax(label)
        if pred_max == target_max:
            number_of_correct += 1

        """
        Back Propagation to adjust weights
        """
        l3.backward_start(label)
        l2.backward_next(l3.output, l3.weight_matrix)
        l1.backward_next(l2.output, l2.weight_matrix)

    print("loop done")
    print(number_of_correct / total_number)
    print("final")

print("Finished...")


