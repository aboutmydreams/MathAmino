import numpy as np
import pandas as pd

from keras.models import load_model,Model
from keras import layers
from acid import amino_acid,value_list

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




# model = load_model('data/model.h5')
#model = load_model('best_models/model7.h5')
model = load_model('test_models/model1.h5')


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


data = np.load('data/counts1.npy')
for k,i in enumerate(to_np().tolist()):
    i1 = np.array([i])
    ans = model.predict_classes(i1)
    print(k,'=====',ans)



#print(data)
dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_1').output)
dense1_output = dense1_layer_model.predict(to_np())

print (dense1_output.shape)

'''
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