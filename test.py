import layer
import data
import numpy as np
import modelParser as mp

filepath = "./model.txt"
print("Running...\n\n")
test_images, test_labels = data.get_test_data()

parsed_file = mp.Parser(filepath, 4)

"""
Initialize Neural layers.
"""
l1 = layer.Layer(784, 784, parsed_file.data[0])
l2 = layer.Layer(784, 784, parsed_file.data[1])
l3 = layer.Layer(784, 784, parsed_file.data[2])
l4 = layer.Layer(784, 10, parsed_file.data[3])

"""
Loops 3 times over the entire dataset of 60000 pictures.
Prints the overall accuracy of our network after each iteration
"""

number_of_correct = 0
total_number = 0
for img, label in zip(test_images, test_labels):

    """
    Forward propagation
    """
    l1.forward(img)
    l2.forward(l1.output)
    l3.forward(l2.output)
    l4.forward(l3.output)

    pred_max = np.argmax(l4.output)
    target_max = np.argmax(label)
    if pred_max == target_max:
        number_of_correct += 1
    total_number += 1
print("FINAL ACCURACY OF ITERATION: " + str((number_of_correct / total_number) * 100) + "%")
print("CORRECT: " + str(number_of_correct))
print("INCORRECT: " + str(total_number - number_of_correct))
print("\n\nFinished...")
