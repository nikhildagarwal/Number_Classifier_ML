import layer
import customData
import numpy as np
import modelParser as mp

filepath = "./model.txt"
print("Running...\n\n")

parsed_file = mp.Parser(filepath, 4)
custom_test_images = []
custom_test_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
imageNames = ["Nikhil_zero",
              "Nikhil_one",
              "Nikhil_two",
              "Nikhil_three",
              "Nikhil_four",
              "Nikhil_five",
              "Nikhil_six",
              "Nikhil_seven",
              "Nikhil_eight",
              "Nikhil_nine"]

for name in imageNames:
    imagePath = "./custom_data/" + name + ".jpg"
    array= customData.extract_custom_data(imagePath)
    custom_test_images.append(array)

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

number_of_correct = 0
total_number = 0
for img, label in zip(custom_test_images, custom_test_labels):

    """
    Forward propagation
    """
    l1.forward(img)
    l2.forward(l1.output)
    l3.forward(l2.output)
    l4.forward(l3.output)

    rounded = []
    for num in l4.output:
        rounded.append(round(num, 2))
    print(rounded)
    print(label)
    pred_max = np.argmax(l4.output)
    if pred_max == label:
        number_of_correct += 1
    total_number += 1
print("FINAL ACCURACY OF ITERATION: " + str((number_of_correct / total_number) * 100) + "%")
print("CORRECT: " + str(number_of_correct))
print("INCORRECT: " + str(total_number - number_of_correct))
print("\n\nFinished...")
