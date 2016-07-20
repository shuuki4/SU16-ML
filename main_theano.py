import csv
import random
import cPickle
import numpy as np
from sklearn import naive_bayes
from sklearn.ensemble.forest import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.linear_model.ridge import Ridge
import sklearn.metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.regression import KNeighborsRegressor
from sklearn import cross_validation
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn import linear_model
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import copy
import time

import theano
import theano.tensor as T
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import *

def feature_generator(row, num_info, min_info, max_info, feature_info_dict) :
	feature_length = np.sum(num_info)
	my_feature = np.zeros((feature_length, ), dtype=np.float32)

	for i in range(33) :
		if i==0 or i==6 : continue

		if num_info[i] == 1 :
			start = int(np.sum(num_info[:i]))
			if row[i] is not "NON" :
				my_feature[start] = (float(row[i])-min_info[i])/(max_info[i] - min_info[i]) 
			else :
				my_feature[start] = 0.5
		else :
			start = int(np.sum(num_info[:i]))
			where = feature_info_dict[i].get(row[i], None)
			if where is not None :
				my_feature[start+where] = 1.0

	return my_feature

def mape(real, predict) :
	real_resc = (real+1.0)/2 * (max_result-min_result) + min_result
	predict_resc = (predict+1.0)/2 * (max_result-min_result) + min_result
	mape_result =  T.abs_(real_resc-predict_resc) / real_resc
	return T.mean(mape_result)

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


filepath_train = 'ech_apprentissage.csv'
filepath_test = 'ech_test.csv'
train_open = open(filepath_train, 'rb')
test_open = open(filepath_test, 'rb')
train_reader = csv.reader(train_open, delimiter=';')
test_reader = csv.reader(test_open, delimiter=';')
header = train_reader.next()
test_reader.next()

train_dataset = []
test_dataset = []
for row in test_reader : 
	test_dataset.append(row)
for row in train_reader : 
	train_dataset.append(row)

for index,row in enumerate(test_dataset):
    test_dataset[index]=[value if value not in ["NR",""] else "NON" for value in row]
for index, row in enumerate(train_dataset) :
    train_dataset[index]=[value if value not in ["NR",""] else "NON" for value in row]

number_info = np.array([0, 1, 1, 154, 1, 1, 1, 5, 1, 1, 17, 1, 1, 2, 1, 2, 5, 1, 23, 1, 1, 1, 1, 4, 8, 1, 6, 6, 1, 1, 2, 2, 1], dtype=np.int32)
print np.sum(number_info)
feature_info_dict = {}
for row in train_dataset :
	for f_i, feature in enumerate(row[:33]) :
		if number_info[f_i] > 1 :
			my_dict = feature_info_dict.get(f_i, {})
			test = my_dict.get(feature, None)
			if test is None :
				my_dict[feature] = len(my_dict)
			feature_info_dict[f_i] = my_dict

min_info = np.zeros((33, ), dtype=np.float32)
max_info = np.zeros((33, ), dtype=np.float32)
min_result = 9999999.99999
max_result = -9999999.99999

for row in train_dataset :
	for i in range(33) :
		if number_info[i] == 1 and i is not 6 :
			if row[i] is not "NON" :
				val = float(row[i])
				if i==1 or i==2 : val = 2016-val
				if val < min_info[i] : min_info[i] = val
				if val > max_info[i] : max_info[i] = val
	result_val = float(row[header.index("prime_tot_ttc")])
	if result_val < min_result : min_result = result_val
	if result_val > max_result : max_result = result_val

print min_result, max_result
max_result = 800

print "info generation done."

train_array = np.zeros((240000, 255))
train_target = np.zeros((240000))
val_array = np.zeros((60000, 255))
val_target = np.zeros((60000, ))
test_array = np.zeros((30000, 255))
test_dataset_ids = []

for index, row in enumerate(train_dataset) :
    if index < 240000 :
    	train_array[index,:] = feature_generator(row, number_info, min_info, max_info, feature_info_dict)
    	train_target[index] = (float(row[header.index("prime_tot_ttc")])-min_result)/(max_result-min_result)*2-1
    else :
    	val_array[index-240000,:] = feature_generator(row, number_info, min_info, max_info, feature_info_dict)
    	val_target[index-240000] = (float(row[header.index("prime_tot_ttc")])-min_result)/(max_result-min_result)*2-1

for index, row in enumerate(test_dataset):
    test_array[index,:] = feature_generator(row, number_info, min_info, max_info, feature_info_dict)
    test_dataset_ids.append(row[0])

print "Start Building Keras Model"

mini_batch_size = 128
nb_epoch = 10

model = Sequential()
model.add(Dense(200, input_shape=(255,)))
model.add(BatchNormalization(epsilon=1e-06, momentum=0.9))
model.add(ELU(alpha=1.0))
model.add(Dense(150))
model.add(BatchNormalization(epsilon=1e-06, momentum=0.9))
model.add(ELU(alpha=1.0))
model.add(Dense(80))
model.add(BatchNormalization(epsilon=1e-06, momentum=0.9))
model.add(ELU(alpha=1.0))
model.add(Dense(30))
model.add(BatchNormalization(epsilon=1e-06, momentum=0.9))
model.add(ELU(alpha=1.0))
model.add(Dense(1))

model.summary()
model.compile(loss=mape, optimizer=RMSprop(lr=0.0002))
#model.load_weights('dnn_weights.h5')
'''
callbacks = [
	EarlyStoppingByLossVal(monitor='val_loss', value=0.108, verbose=1)
]
'''

history = model.fit(train_array, train_target, batch_size=mini_batch_size, nb_epoch=300, verbose=2)
last_result = model.predict(test_array, batch_size=mini_batch_size)

with open('result_keras3.csv', 'wb') as csvfile :

    fieldnames = ['ID', 'COTIS']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

    writer.writeheader()
    for index, value in enumerate(last_result) :
    	writer.writerow({'ID' : test_dataset_ids[index], 'COTIS' : (np.squeeze(value)+1.0)/2*(max_result-min_result)+min_result})

model.save_weights('dnn_weights_2.h5')