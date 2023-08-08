# Number_Classifier_ML
Hi! Welcome to my ML Number Classifier Project. Enjoy!!

I have built a neural network from scratch in python making use of numpy and Object-Oriented-Programming.
This network takes a 28x28 pixel image of a handwritten digit and correctly classifies the digit as 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9

This Project is written in Python and requires the use of numpy, pillow, math, and pathlib.
You can install these via an IDE or with "pip install" in the command line.

**SUMMARY**:
* Utilizes machine learning algorithms/concepts like forward/back propagation, activation functions, and loss
calculation for intelligent learning.
* Trained the network on 60,000 images, which allowed the network to correctly classify 97.64% of 10,000 new
images it had never seen before.
* The 3 main files to run are: **train.py**, **test.py**, and **customTest.py**

**train.py**:
* Trains the neural network with images extracted from the .npz file in the data folder.
* Checks to first see if model.txt currently has data.
     + If it does, the script will load the model weights into the network and start training based on the provided model weights.
     + Otherwise, the weights between the layers in our network will be initialized with random weights and trained from scratch.
     + Clear the model.txt file to retrain the network from the beginning
* Forward propogates pixel data through the network while implementing an Activation function at the end of the each neuron.
* Calculates loss at the end of the network based on the correct answer provided by training data, and back propogates with a learning rate of 0.01
* Prints the learning progress of the network in the console.

**test.py**:
* Extracts different images from .npz file in the data folder than train.py
    + These images are new images that the network has never seen before
* Loads the model.txt weights into the neural network if the data exists. If the data doesn't exist, the script will run the test images through a randomized network
    + Running test images through randomized network shows that my ML algorithms and neural network actually work!!
* At conclusion, script prints Accuracy of the model

**customTest.py**:
* Does the same as test.py, HOWEVER, uses custom images of my own handwritten digits





