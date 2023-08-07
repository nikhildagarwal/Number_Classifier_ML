import numpy as np


class Parser:
    """
    Parser Object to extract all weights from text file and build neural network.
    Allows us to train on top of the previous training as well replicate the model whenever necessary witout
    re-training the network.
    """
    def __init__(self, filepath, length):
        """
        Initializes parser with filepath to model text file, and the number of layers that we want to extract from the file.
        :param filepath: file path to model text file
        :param length: number of layers
        """
        self.data = []
        for i in range(length):
            self.data.append(None)
        with open(filepath, 'r') as file:
            content = file.read()
        if content != "":
            matrix_strings = content.split('%')
            counter = 0
            for ms in matrix_strings:
                two_d = []
                truncate = ms[1:len(ms)-1]
                rows = truncate.split("][")
                for row in rows:
                    one_d = []
                    row.replace("\n", "")
                    nums = row.split(" ")
                    for num in nums:
                        try:
                            one_d.append(float(num))
                        except ValueError:
                            pass
                    two_d.append(one_d)
                self.data[counter] = np.asarray(two_d)
                counter += 1
