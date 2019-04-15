import numpy as np
import pandas as pd
import os
from keras.models import load_model,Model
from keras import layers
from acid import amino_acid,value_list
import matplotlib.pyplot as plt

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

all_need = []
for acid in need_pre:
    this_acid = [acid]
    s_index = [k for k,v in enumerate(list(acid)) if v=='S']
    s_index.remove(7)
    print(s_index)
    if s_index != []:
        for i in s_index:
            if i > 7:
                another = acid[i-7:]+'O'*(i-7)
                this_acid.append(another)
            if i < 7:
                another = 'O'*(7-i)+acid[:i+7]
                this_acid.append(another)
    all_need.append(this_acid)
# print(all_need)





def to_np():
    acids_counts = []

    for acid in need_pre:

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
    # acid_pd_count.to_csv('data/test_data.csv')
    np_count = np.array(acid_pd_count)
    return np_count

def to_np2():
    def good_num(num):
        go_num = (num-0.5)*2
        return go_num

    all_counts = []
    
    for acids in all_need:
        acids_counts = []
        for acid in acids:
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

        column_names = list(value_list.keys())
        column_names.extend(list(range(15)))
        column_names.extend(['v1','v2','v3','v4','v5','v6','v7'])

        acid_pd_count = pd.DataFrame(acids_counts,columns=column_names)
        acid_pd_count = acid_pd_count.astype('float')
        # acid_pd_count.to_csv('data/test_data.csv')
        np_count = np.array(acid_pd_count)
        all_counts.append(np_count)
    return all_counts

# 加载模型
# model = load_model('data/model.h5')
# model = load_model('best_models/model700.h5')
# model = load_model('test_models/model10.h5')


# json_string = model.to_json()
# weight_Dense_1,bias_Dense_1 = model.get_layer('Dense').get_weights()
# print(weight_Dense_1.shape)
# print(bias_Dense_1.shape)

# open('my_model_architecture.json','w').write(json_string)
# model = model_from_json(open('my_model_architecture.json').read())  
# model.load_weights('my_model_weights.h5') 
# a = model.get_config()
# print("model config is ",a)

# model = load_model('data/model.h5')


# for i in need_pre:
#     aaa = amino_acid(i)
#     counts = np.array([aaa.counts]).astype('float')
#     counts = counts.reshape(counts.shape[0],-1)
#     # print(np.array(counts))
#     ans = model.predict_classes(counts)
#     print(ans)


# test 1
'''
data = np.load('data/mydata.npy')
for k,i in enumerate(data.tolist()):
    aaa = amino_acid(i)
    counts = np.array([aaa.counts]).astype('float')
    counts = counts.reshape(counts.shape[0],-1)
    # print(np.array(counts))
    ans = model.predict_classes(counts)
    print(k,'=====',ans)
print(data)
'''


def new_pre():
    need_data = pd.read_csv('data/need_pre.csv')
    need_data = need_data.drop(['Unnamed: 0'],axis=1)
    print(need_data.style)
    need_data = need_data.values
    the_acids = need_data[:10]
    # print(the_acids)
    return the_acids


'''
data = np.load('data/counts3.npy')
def test_pre(model):
    last_ans = []
    model0 = load_model(model)
    for k,one_acid in enumerate(to_np2()):
        ans_list = []
        for i in one_acid:
            one_acid1 = np.array([i])
            ans = model0.predict_classes(one_acid1)
            print(ans)
            ans_list.append(ans)
        if 1 in ans_list:
            my_ans = 1
        else:
            my_ans = 0
        print(k,'=====',my_ans)
        last_ans.append(my_ans)
    return last_ans
'''

# data_list = os.listdir('train_models')

# path_name = '/Users/dwh/Downloads/mytrain/'
path_name = 'train_models/'
data_list = os.listdir(path_name)

all_ans = []
name_list = []
for file in data_list:
    # if file != '.DS_Store':
    if str(file) == '0.77698.h5':
        name = path_name + file
        model = load_model(name)
        np_data = np.array(new_pre())
        answer = model.predict_classes(np_data)
        aaa  = model.predict(np_data)
        print(aaa)
        # answer = np.array(answer)
        # acc = (answer==np.array([1,0,0,0,1,0,1,1,0,1]))
        all_ans.append(answer.tolist())
        name_list.append(name)
        # print(model.summary())
        # print(model.get_config())

        dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_1').output)
        dense1_output = dense1_layer_model.predict(np_data)
        # print(dense1_output)
        # print (dense1_output.shape)


        #获得某一层的权重和偏置
        # weight_Dense_1,bias_Dense_1 = model.get_layer('Dense_1').get_weights()
        # print(weight_Dense_1.shape,bias_Dense_1.shape)
        # print(weight_Dense_1)
        # print(bias_Dense_1)

        # weight_Dense_2,bias_Dense_2 = model.get_layer('Dense_2').get_weights()
        # print(weight_Dense_2.shape,bias_Dense_2.shape)
        # print(weight_Dense_2)
        # print(bias_Dense_2)

        # weight_Dense_3,bias_Dense_3 = model.get_layer('Dense_3').get_weights()
        # print(weight_Dense_3.shape,bias_Dense_3.shape)
        # print(weight_Dense_3)
        # print(bias_Dense_3)

        print(np.array(all_ans))
        print(name_list)



'''
#print(data)
dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_1').output)
dense1_output = dense1_layer_model.predict(to_np())

print (dense1_output.shape)


#获得某一层的权重和偏置
weight_Dense_1,bias_Dense_1 = model.get_layer('Dense_1').get_weights()
print(weight_Dense_1.shape,bias_Dense_1.shape)
print(weight_Dense_1)
print(bias_Dense_1)

weight_Dense_2,bias_Dense_2 = model.get_layer('Dense_2').get_weights()
print(weight_Dense_2.shape,bias_Dense_2.shape)
print(weight_Dense_2)
print(bias_Dense_2)

weight_Dense_3,bias_Dense_3 = model.get_layer('Dense_3').get_weights()
print(weight_Dense_3.shape,bias_Dense_3.shape)
print(weight_Dense_3)
print(bias_Dense_3)
'''