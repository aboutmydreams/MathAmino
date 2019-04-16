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
        self.counts = self.count_it(str_data)
        self.mean_valueS1 = self.near_value(str_data,value_list,near_n=1,no_S=0).mean()
        self.mean_value1 = self.near_value(str_data,value_list,near_n=1,no_S=1).mean()
        self.mean_valueS2 = self.near_value(str_data,value_list,near_n=2,no_S=0).mean()
        self.mean_value2 = self.near_value(str_data,value_list,near_n=2,no_S=1).mean()
        self.mean_valueS3 = self.near_value(str_data,value_list,near_n=3,no_S=0).mean()
        self.mean_value3 = self.near_value(str_data,value_list,near_n=3,no_S=1).mean()
        self.mean_valueS4 = self.near_value(str_data,value_list,near_n=4,no_S=0).mean()
        self.mean_value4 = self.near_value(str_data,value_list,near_n=4,no_S=1).mean()
        self.mean_valueS5 = self.near_value(str_data,value_list,near_n=5,no_S=0).mean()
        self.mean_value5 = self.near_value(str_data,value_list,near_n=5,no_S=1).mean()
        self.mean_value6 = self.near_value(str_data,value_list,near_n=6,no_S=1).mean()
        self.mean_value7 = self.near_value(str_data,value_list,near_n=7,no_S=1).mean()

    def to_value(self, str_data):
        row_mode = []
        for i in list(str_data):
            value = value_list[i]
            row_mode.append(value)
        row_mode = np.array(row_mode)
        return row_mode

    def count_it(self,str_data):
        c = Counter(str_data)
        count_list = []
        for k in list(value_list.keys()): #['L','D','F','C','P','S','W','Y','I','V','O']:
            count_list.append(c[k])
        return count_list

    def near_value(self,str_data,data_list,near_n,no_S=1):
        vas = []
        for one_str in list(str_data[7-near_n:8+near_n]):
            va = data_list[one_str]
            vas.append(va)
        if no_S==1:
            length = len(vas)
            S_site = length//2
            vas.pop(S_site)
        
        def dele_zero(li):
            if 0 in li:
                li.remove(0)
                return dele_zero(li)
            else:
                return li
        
        vas = dele_zero(vas)
        return np.array(vas)

        


