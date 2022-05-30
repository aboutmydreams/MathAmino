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

# test all data

#[7,70,72,73,79,700,708,729]

for i in range(10):
    model = load_model(f'test_models/model{str(i)}.h5')
    test_train_data = np.load('data/counts1.npy')
    aims = pd.read_csv('data/mydata.csv')
    x_test = test_train_data
    y_test = aims.aims.tolist()
    y_test = np_utils.to_categorical(y_test,num_classes=2)
    loss,accuracy = model.evaluate(x_test,y_test)
    print('loss:',loss)
    print('accuracy:',accuracy)

