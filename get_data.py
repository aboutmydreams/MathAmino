# -*- coding:utf-8 -*-

import xlrd
import numpy as np
import pandas as pd
from xlutils.copy import copy
from acid import amino_acid


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





# 保存data npy
'''
data = xlrd.open_workbook('data/Data.xlsx')
then_excel = copy(data)
sheet0 = data.sheets()[0]
this_data = []

for i in range(1,401):
    row_data = sheet0.row_values(i)[0]
    this_data.append(row_data)
    this_acid = amino_acid(row_data)
    num_list = this_acid.value.tolist()
    then_excel.get_sheet(0).write(i,0,str(num_list))

then_excel.save('data/thendata.xlsx')
this_data = np.array(this_data)
np.save("data/mydata.npy",this_data)
'''


# 验证数据的正确性
'''

# print(len(this_data),this_data[0])
# all_keys = list(value_list.keys())
# aaa = ''.join(this_data)
# for k,i in enumerate(list(aaa)):
#     if i not in all_keys:
#         print(k,'--',i) # no print

'''



# 保存原始csv

datas = np.load('data/mydata.npy')
acids = datas.tolist()
aims = np.ones(200).tolist() + np.zeros(200).tolist()
acids_datas = {'acids': acids,'aims': aims}
original_csv = pd.DataFrame(acids_datas)
# doriginal_csv.to_csv('mydata.csv')
# original_csv = pd.read_csv('data/mydata.csv')
acids_counts = []

for acid in list(original_csv['acids']):

    one_acid = amino_acid(acid)
    counts = one_acid.counts
    counts.extend(one_acid.value.tolist())
    counts.append(one_acid.mean_valueS1)
    counts.append(one_acid.mean_value1)
    counts.append(one_acid.mean_valueS2)
    counts.append(one_acid.mean_value2)
    counts.append(one_acid.mean_valueS3)
    counts.append(one_acid.mean_value3)
    counts.append(one_acid.mean_valueS4)
    counts.append(one_acid.mean_value4)
    counts.append(one_acid.mean_valueS5)
    counts.append(one_acid.mean_value5)

    acids_counts.append(counts)

column_names = list(value_list.keys())
column_names.extend(list(range(15)))
column_names.extend(['vs1','v1','vs2','v2','vs3','v3','vs4','v4','vs5','v5'])

acid_pd_count = pd.DataFrame(acids_counts,columns=column_names)
acid_pd_count = acid_pd_count.astype('float')
acid_pd_count.to_csv('data/test_data.csv')
np_count = np.array(acid_pd_count)
print(np_count)
np.save('data/counts1.npy',np_count)
# original_csv = original_csv.join(acid_pd_count)
print(acid_pd_count.head(5))
# original_csv.to_csv('data/mydata.csv')
