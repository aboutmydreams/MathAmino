import numpy as np
import pandas as pd
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import load_model
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from showimg import LossHistory
from do_model import test_pre


test_train_data = np.load('data/counts1.npy')
aims = pd.read_csv('data/mydata.csv')

def get_best(num):
    # history = LossHistory()

    tensorboard = TensorBoard(log_dir='log/')
    callback_lists = [tensorboard]  #因为callback是list型,必须转化为list

    # print(aims.aims)
    x_train, x_test, y_train, y_test = train_test_split(test_train_data,aims.aims,test_size=0.10)
    y_train, y_test = y_train.tolist(),y_test.tolist()
    y_train = np_utils.to_categorical(y_train,num_classes=2)
    y_test = np_utils.to_categorical(y_test,num_classes=2)
    # print(x_train[0])

    # 模型
    model = Sequential()
    # 去拟合训练
    model.add(Dense(units=16,input_dim=46,bias_initializer='one',activation='sigmoid',name="Dense_1"))
    # bias_initializer='one',activation='tanh'


    model.add(Dropout(0.2))
    model.add(Dense(units=8,activation='relu'))#activation='relu'
    model.add(Dropout(0.2))
    model.add(Dense(units=2,activation='softmax'))

    # model.summary()


    # 损失函数使用交叉熵
    sgd = SGD(lr=0.003)
    model.compile(loss='categorical_crossentropy',#'binary_crossentropy''categorical_crossentropy'
            optimizer='rmsprop',  # sgd, # 'rmsprop'
            metrics=['accuracy'])
    #模型估计
    model.fit(x_train, y_train, epochs=200, batch_size=25,callbacks=callback_lists)#verbose='True',callbacks=callback_lists)# callbacks=[history])


    loss,accuracy = model.evaluate(x_train, y_train)#(x_train, y_train)#(x_test,y_test)
    print('loss:',loss)
    print('accuracy:',accuracy)
    loss1,accuracy1 = model.evaluate(x_test,y_test)
    print('loss:',loss1)
    print('accuracy:',accuracy1)


    plot_model(model, to_file='model1.png',show_shapes=True)
    # history.loss_plot('epoch')
    model.save(f'test_models/model{str(num)}.h5')
    logs = open('test_models/logs.txt','a')
    logs.write(
        f'{str(num)}=={str(accuracy1)}||{str(accuracy)}||{str(loss1)}||{str(loss)}'
        + '--------------\n'
    )

def train_best(model0,epoch=200,num=1):
    tensorboard = TensorBoard(log_dir='log/')
    callback_lists = [tensorboard]  #因为callback是list型,必须转化为list

    x_train, x_test, y_train, y_test = train_test_split(test_train_data,aims.aims,test_size=0.10)
    y_train, y_test = y_train.tolist(),y_test.tolist()
    y_train = np_utils.to_categorical(y_train,num_classes=2)
    y_test = np_utils.to_categorical(y_test,num_classes=2)

    # 训练完成 导入模型
    model = load_model(model0)

    # 损失函数使用交叉熵
    sgd = SGD(lr=0.003)
    model.compile(loss='binary_crossentropy',#'binary_crossentropy''categorical_crossentropy'
            optimizer='rmsprop',  # sgd, # 'rmsprop'
            metrics=['accuracy'])
    #模型估计
    model.fit(x_train, y_train, epochs=epoch, batch_size=25,callbacks=callback_lists)#verbose='True',callbacks=callback_lists)# callbacks=[history])

    loss,accuracy = model.evaluate(x_train, y_train)#(x_train, y_train)#(x_test,y_test)
    print('loss:',loss)
    print('accuracy:',accuracy)
    loss1,accuracy1 = model.evaluate(x_test,y_test)
    print('loss:',loss1)
    print('accuracy:',accuracy1)


    model.save(f'train_models/{str(num)}.h5')
    # logs = open('test_models/logs.txt','a')
    # logs.write(str(num) + '==' + str(accuracy1) + '||' + str(accuracy) + '||' + str(loss1) + '||' + str(loss) + '--------------\n')

# for i in range(10):
#     get_best(i)
    # train_best(model0='test_models/model10.h5',epoch=200,num=i)

for i in range(10):
    test_pre(f'test_models/model{str(i)}.h5')
    print('---'*10)