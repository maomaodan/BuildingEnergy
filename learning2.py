import scipy.io as sio
import tensorflow as tf
import numpy as np
import sys
import datetime
import os
import pandas as pd

# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# from keras.layers.recurrent import LSTM
# from keras.layers.wrappers import TimeDistributed
# from keras.layers.convolutional import Convolution1D
# from keras.layers.pooling import MaxPooling1D
# from keras.layers.core import Flatten
# from keras.engine.topology import Merge
# from keras import backend as K
# from keras.optimizers import SGD
# from keras.objectives import *
# from keras.utils.layer_utils import layer_from_config
# from keras.regularizers import l1, l2
# from keras.callbacks import *


def load_data():
	read = np.loadtxt(open('minute.csv'),delimiter=',')	
	label = read[1:,1]
	data = read[:-1,[0,2,3,4,5,6,7,8,9,10,11]]
	return data, label

def NLL(pred, truth):
	dist = tf.contrib.distributions.Normal(mu=pred[0], sigma=tf.abs(pred[1]))
	return tf.reduce_mean(-dist.log_pdf(truth))

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# K.set_session(session)

data, label = load_data()
print data
print label

# tf.python.control_flow_ops = tf

# sgd = SGD(lr=0.01, momentum=0.9, decay=0.01, nesterov=True)

# early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')

# model = Sequential()

# model.add(Dense(128, activation='relu', init='normal', input_dim = 11))

# model.add(Dense(128, activation='relu', init='normal'))

# model.add(Dense(128, activation='relu', init='normal'))
# model.add(Dense(2, init='normal'))
# model.compile(loss=NLL, optimizer=sgd)

x_train = data[:len(data)/3 * 2]
y_train = label[:len(label)/3 * 2]

x_test = data[len(data)/3 * 2:]
y_test = label[len(label)/3 * 2:]

# history = model.fit(x_train, y_train, nb_epoch=300, batch_size=256, verbose=1)
# y_pred = model.predict(x_test, verbose=0)

from sklearn.linear_model import LinearRegression

clf = LinearRegression()

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

import matplotlib.pyplot as plt
plt.clf()
plt.plot(y_pred[:], label='pred(mean)')
# plt.plot(y_pred[:,1], label='pred(std)')
plt.plot(y_test, label='truth')
plt.legend()
plt.show()

mse = np.mean((y_pred - y_test) ** 2)
print mse
