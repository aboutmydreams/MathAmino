import numpy as np
from collections import Counter

aim_amino_acid = ['S','T','Y']

value_list = {
    'A': 0.934,
    'L': 0.825,
    'R': 0.962,
    'K': 1.04,
    'N': 0.986,
    'M': 0.804,
    'D': 0.994,
    'F': 0.773,
    'C': 0.9,
    'P': 1.047,
    'Q': 1.047,
    'S': 1.056,
    'E': 0.986,
    'T': 1.008,
    'G': 1.015,
    'W': 0.848,
    'H': 0.882,
    'Y': 0.931,
    'I': 0.766,
    'V': 0.825,
    'O': 0,
}


class amino_acid:
    def __init__(self, str_data):
        self.value = self.to_value(str_data)


    def to_value(self, str_data):
        row_mode = []
        for i in list(str_data):
            value = value_list[i]
            row_mode.append(value)
        row_mode = np.array(row_mode)
        return row_mode

    def cont_it(self,str_data):
        
