# -*- coding:utf-8 -*-

import xlrd
import numpy as np
import pandas as pd
from xlutils.copy import copy
from acid import amino_acid,value_list
from sklearn.preprocessing import MinMaxScaler


aim_amino_acid = ['S','T','Y']

need_pre = [
    'GKKSGHSSPGAIIVS',
    'TSANEHTSAVOOOOO',
    'LAVYPFASLPEEIPR',
    'MKLTDLLSLINSTHL',
    'RKPARVYSVSSDIVP',
    'PKVNLVKSEGYVTDG',
    'PFRVISVSSNSNSRN',
    'TNPFRVISVSSNSNS',
    'TRMNSFYSNILIVGG',
    'KSSLSLNSDLSTPHF',
]


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

def good_num(num):
    go_num = (num-0.5)*2
    return go_num


# 保存原始csv
def to_counts2():
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
        v_lists = [one_acid.mean_value1,\
            one_acid.mean_value2,one_acid.mean_value3,\
            one_acid.mean_value4,one_acid.mean_value5,\
            one_acid.mean_value6,one_acid.mean_value7]
        # counts.append(one_acid.mean_valueS1)
        v_lists = map(good_num,v_lists)
        counts.extend(v_lists)
        acids_counts.append(counts)

    column_names = list(value_list.keys()) #['L','D','F','C','P','S','W','Y','I','V','O']
    column_names.extend(list(range(15)))
    column_names.extend(['v1','v2','v3','v4','v5','v6','v7'])

    acid_pd_count = pd.DataFrame(acids_counts,columns=column_names)
    acid_pd_count = acid_pd_count.astype('float')

    # 保存变量
    acid_pd_count.to_csv('data/test_data3.csv')
    np_count = np.array(acid_pd_count)
    # print(np_count)

    np.save('data/counts3.npy',np_count)
    # original_csv = original_csv.join(acid_pd_count)
    print(acid_pd_count.head(5))
    # original_csv.to_csv('data/mydata.csv')

# to_counts2()

def add_need():
    datas = np.load('data/mydata.npy')
    acids = datas.tolist()
    aims = np.ones(200).tolist() + np.zeros(200).tolist()
    acids_datas = {'acids': acids,'aims': aims}
    original_csv = pd.DataFrame(acids_datas)

    acids_counts = []

    need_pre.extend(list(original_csv['acids']))
    for acid in need_pre:

        one_acid = amino_acid(acid)
        counts = one_acid.counts
        counts.extend(one_acid.value.tolist())
        v_lists = [one_acid.mean_value1,\
            one_acid.mean_value2,one_acid.mean_value3,\
            one_acid.mean_value4,one_acid.mean_value5,\
            one_acid.mean_value6,one_acid.mean_value7]
        # counts.append(one_acid.mean_valueS1)
        v_lists = map(good_num,v_lists)
        counts.extend(v_lists)
        acids_counts.append(counts)

    column_names = list(value_list.keys()) #['L','D','F','C','P','S','W','Y','I','V','O']
    column_names.extend(list(range(15)))
    column_names.extend(['v1','v2','v3','v4','v5','v6','v7'])

    acid_pd_count = pd.DataFrame(acids_counts,columns=column_names)
    acid_pd_count = acid_pd_count.astype('float')

    # 保存变量
    acid_pd_count.to_csv('data/need_pre.csv')
    np_count = np.array(acid_pd_count)
    # print(np_count)

    np.save('data/need_pre.npy',np_count)
    # original_csv = original_csv.join(acid_pd_count)
    print(acid_pd_count.head(15))
