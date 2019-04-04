# -*- coding:utf-8 -*-

import xlrd
import numpy as np
import pandas as pd
from xlutils.copy import copy
from acid import aim_amino_acid


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





# 保存data
'''
data = xlrd.open_workbook('Data.xlsx')
then_excel = copy(data)
sheet0 = data.sheets()[0]
this_data = []

for i in range(1,401):
    row_data = sheet0.row_values(i)[0]
    this_data.append(row_data)
    this_acid = amino_acid(row_data)
    num_list = this_acid.value.tolist()
    then_excel.get_sheet(0).write(i,0,str(num_list))

then_excel.save('thendata.xlsx')
this_data = np.array(this_data)
np.save("mydata.npy",this_data)

# 验证数据的正确性

# print(len(this_data),this_data[0])
# all_keys = list(value_list.keys())
# aaa = ''.join(this_data)
# for k,i in enumerate(list(aaa)):
#     if i not in all_keys:
#         print(k,'--',i) # no print

'''

