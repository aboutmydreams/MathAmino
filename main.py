import numpy as np
import pandas as pd
import random
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout,LSTM
from keras.optimizers import SGD
from keras.models import load_model
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.utils.vis_utils import plot_model
from keras.initializers import RandomNormal,random_normal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from showimg import LossHistory
history = LossHistory()

# tensorboard = TensorBoard(log_dir='log/')
# callback_lists = [tensorboard]  #因为callback是list型,必须转化为list

test_train_data = np.load('data/counts3.npy')
scaler = MinMaxScaler(feature_range=(-1, 1))
test_train_data = scaler.fit_transform(test_train_data)

print(test_train_data)
aims = pd.read_csv('data/mydata.csv')

# print(test_train_data)


# print(x_train[0])


# 训练完成 导入模型
#model = load_model('data/model.h5')
#model = load_model('model0.h5')




def train_model():
    x_train, x_test, y_train, y_test = train_test_split(test_train_data,aims.aims,test_size=0.2)
    y_train, y_test = y_train.tolist(),y_test.tolist()
    y_train = np_utils.to_categorical(y_train,num_classes=2)
    y_test = np_utils.to_categorical(y_test,num_classes=2)


    # x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    # x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    # 模型
    model = Sequential()
    # 全连接模型测试
    # model.add(Dense(units=2,input_dim=46,bias_initializer='one',activation='softmax'))

    b_initializer = RandomNormal(mean=1,stddev=0.1)
    # 去拟合训练
    model.add(Dense(units=22,input_dim=43,
    kernel_initializer=random_normal(stddev=0.01),
    bias_initializer=b_initializer,
    activation='tanh',
    name="Dense_1"))
    # bias_initializer='one',activation='tanh','sigmoid'
    # model.add(LSTM(20, input_shape=(x_train.shape[1], x_train.shape[2]),activation='sigmoid'))

    model.add(Dropout(0.2))
    model.add(Dense(units=12,activation='elu',name="Dense_2"))#activation='relu'
    model.add(Dropout(0.2))
    model.add(Dense(units=2,activation='sigmoid',name="Dense_3"))

    # model.summary()


    # 损失函数使用交叉熵
    # sgd = SGD(lr=0.005)
    model.compile(loss='binary_crossentropy',#'binary_crossentropy''categorical_crossentropy'
            optimizer='AMSGrad',  # sgd, # 'rmsprop' 'adam'
            metrics=['accuracy'])
    #模型估计
    model.fit(x_train, y_train, epochs=200, batch_size=50,
    callbacks=[history],verbose=0)#verbose='True',callbacks=callback_lists)# callbacks=[history])


    loss,accuracy = model.evaluate(x_train, y_train)#(x_train, y_train)#(x_test,y_test)
    loss1,accuracy1 = model.evaluate(x_test,y_test)
    # print('loss:',loss)
    print('accuracy:',accuracy)
    # print('loss:',loss1)
    print('accuracy:',accuracy1)



    plot_model(model,to_file='model1.png',show_shapes=True)
    history.loss_plot('epoch')
    if accuracy1 >= 0.7:
        rd = str(random.randint(1000,9999))
        model.save(f'train_models/{str(accuracy1) + rd}.h5')
        log = open('log.txt','a')
        log.write(str(accuracy1) + rd + '--' + str(accuracy) + '\n')   

for _ in range(1):
    train_model()

#--------------
# dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_1').output)
# dense1_output = dense1_layer_model.predict(x_test[0:5])

# print (dense1_output.shape)


#获得某一层的权重和偏置
# weight_Dense_1,bias_Dense_1 = model.get_layer('Dense_1').get_weights()
# print(weight_Dense_1.shape,bias_Dense_1.shape)
# print(weight_Dense_1)
# print(bias_Dense_1)
