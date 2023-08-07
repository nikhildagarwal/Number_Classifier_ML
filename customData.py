from PIL import Image
import numpy as np


def extract_custom_data(imagepath):
    """
    function to return array of length 784
    :param imagepath: path to custom image
    :return: numpy array of pixel values and integer answer
    """
    image = Image.open(imagepath)
    imagearray = np.array(image)
    toReturn = []
    for row in imagearray:
        for num in row:
            toAdd = (num * -1) + 255
            toReturn.append(toAdd / 255)
    return np.array(toReturn)

