import scipy.io as sio
import tensorflow as tf
import numpy as np
import sys
import datetime
import os
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Flatten
from keras.engine.topology import Merge
from keras import backend as K
from keras.optimizers import SGD
from keras.objectives import *
from keras.utils.layer_utils import layer_from_config
from keras.regularizers import l1, l2
from keras.callbacks import *

def stride(input_x, window_size):
	nrows = input_x.shape[0] - window_size + 1
	p,q = input_x.shape
	m,n = input_x.strides
	strided = np.lib.stride_tricks.as_strided
	output_x = strided(input_x,shape=(nrows,window_size,q),strides=(m,m,n))
	print output_x.shape
	# output_y = input_y[window_size-1:]
	# print output_x.shape, output_y.shape
	# assert len(output_x) == len(output_y)
	return output_x#, output_y

def load_data():
	read = np.loadtxt(open('minute.csv'),delimiter=',')	
	# label = np.loadtxt(open('label.csv'),delimiter=',')	
	label = read[:,1]
	data = read[:,[0,2,3,4,5,6,7,8,9,10,11]]
	# data = stride(data, window)
	# label = label[window-1:]
	return data, label


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

#column_name = ['', 'bathroom', 'kitchen', 'bedroom'][column_id]

#print "#" * 10
#print column_id, column_name

data, label = load_data()
print data
print label
'''
for i in range(len(data[0])):
	data[:,i] -= np.mean(data[:,i])
	data[:,i] /= np.std(data[:,i])
'''
#label = np.vstack((label, 1-label)).T

#data = data#[:-1]
#label = label#[1:]

tf.python.control_flow_ops = tf

sgd = SGD(lr=0.0001, momentum=0.9, decay=0.01, nesterov=True)
#csv_logger = CSVLogger('csv/loss@%s.csv'%column_name)
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')

model = Sequential()

# model.add(TimeDistributed(Dense(128, activation='relu', init='normal'), input_shape=(12, 11)))
# model.add(LSTM(128, consume_less='gpu'))
# model.add(Dropout(0.5))
# model.add(TimeDistributed(Dense(32, activation='relu', init='normal')))
model.add(Dense(128, activation='relu', init='normal', input_shape = (12,11)))
# model.add(Dropout(0.5))
model.add(Dense(128, activation='tanh', init='normal'))
# model.add(Dropout(0.5))
model.add(Dense(128, activation='tanh', init='normal'))
model.add(Dense(2, activation='softmax', init='normal'))
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

x_train = data[:len(data)/3 * 2]
y_train = label[:len(label)/3 * 2]

x_test = data[len(data)/3 * 2:]
y_test = label[len(label)/3 * 2:]

#positive = np.sum(y_train[:, 0])
#negative = np.sum(y_train[:, 1])

#sample_weight = y_train[:, 0] * negative + y_train[:, 1] * positive

# print positive, negative
# exit()

history = model.fit(x_train, y_train, nb_epoch=10000, batch_size=256, verbose=0)
y_pred = model.predict(x_test, verbose=0)
'''
TP = 0
TN = 0
FP = 0
FN = 0

P = 0

P = np.sum(y_test[:, 0])
TP = np.sum(np.round(y_pred[:,0]) * y_test[:, 0])
TN = np.sum(np.round(y_pred[:,1]) * y_test[:, 1])

FP = np.sum(np.round(y_pred[:,0]) * y_test[:, 1])
FN = np.sum(np.round(y_pred[:,1]) * y_test[:, 0])

print "TP: ", TP
print "TN: ", TN
print "FP: ", FP
print "FN: ", FN
'''
# print y_pred
