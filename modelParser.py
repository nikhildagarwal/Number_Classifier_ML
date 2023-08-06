import numpy as np


class Parser:

    def __init__(self, filepath, length):
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
