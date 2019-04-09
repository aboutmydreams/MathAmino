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


def get_best(num):
    # history = LossHistory()

    tensorboard = TensorBoard(log_dir='log/')
    callback_lists = [tensorboard]  #因为callback是list型,必须转化为list

    test_train_data = np.load('data/counts1.npy')
    aims = pd.read_csv('data/mydata.csv')
    # print(aims.aims)
    x_train, x_test, y_train, y_test = train_test_split(test_train_data,aims.aims,test_size=0.10)
    y_train, y_test = y_train.tolist(),y_test.tolist()
    y_train = np_utils.to_categorical(y_train,num_classes=2)
    y_test = np_utils.to_categorical(y_test,num_classes=2)
    # print(x_train[0])


    # 训练完成 导入模型
    #model = load_model('data/model.h5')
    #model = load_model('model0.h5')

    # 模型
    model = Sequential()
    # 全连接模型测试
    # model.add(Dense(units=2,input_dim=46,bias_initializer='one',activation='softmax'))

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
    model.compile(loss='binary_crossentropy',#'binary_crossentropy''categorical_crossentropy'
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
    model.save('test_models/model{}.h5'.format(str(num)))
    logs = open('test_models/logs.txt','a')
    logs.write(str(num) + '==' + str(accuracy1) + '||' + str(accuracy) + '||' + str(loss1) + '||' + str(loss) + '--------------\n')
    #--------------
    # dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_1').output)
    # dense1_output = dense1_layer_model.predict(x_test[0:5])

    # print (dense1_output.shape)


    #获得某一层的权重和偏置
    # weight_Dense_1,bias_Dense_1 = model.get_layer('Dense_1').get_weights()
    # print(weight_Dense_1.shape,bias_Dense_1.shape)
    # print(weight_Dense_1)
    # print(bias_Dense_1)


for i in range(10):
    get_best(i)