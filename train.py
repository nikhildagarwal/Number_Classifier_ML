import layer
import data
import numpy as np
import modelParser as mp

filepath = "./model.txt"
print("Running...\n\n")
train_images, train_labels = data.get_training_data()

parsed_file = mp.Parser(filepath, 4)

"""
Initialize Neural layers.
"""
l1 = layer.Layer(784, 392, parsed_file.data[0])
l2 = layer.Layer(392, 196, parsed_file.data[1])
l3 = layer.Layer(196, 98, parsed_file.data[2])
l4 = layer.Layer(98, 10, parsed_file.data[3])

"""
Loops 3 times over the entire dataset of 60000 pictures.
Prints the overall accuracy of our network after each iteration
"""
for i in range(3):
    number_of_correct = 0
    total_number = 0
    counter = 0
    for img, label in zip(train_images, train_labels):

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
        if counter % 100 == 0:
            print("running accuracy: " + str((number_of_correct / total_number) * 100) + "%")
        counter += 1

        """
        Back Propagation to adjust weights
        """
        l4.backward_start(label)
        l3.backward_next(l4.output, l4.weight_matrix)
        l2.backward_next(l3.output, l3.weight_matrix)
        l1.backward_next(l2.output, l2.weight_matrix)

    print("FINAL ACCURACY OF ITERATION " + str(i) + ": " + str((number_of_correct / total_number) * 100) + "%")

file_mode = "w"
content_to_write = ""
for n in l1.weight_matrix:
    content_to_write += str(n)
content_to_write += "%"
for n in l2.weight_matrix:
    content_to_write += str(n)
content_to_write += "%"
for n in l3.weight_matrix:
    content_to_write += str(n)
content_to_write += "%"
for n in l4.weight_matrix:
    content_to_write += str(n)
with open(filepath, file_mode) as file:
    file.write(content_to_write)

print("\n\nFinished...")
