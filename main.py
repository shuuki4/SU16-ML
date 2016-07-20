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
from scipy.interpolate import UnivariateSpline
import copy

import xgboost as xgb

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
	error = 0.0
	for i in range(real.shape[0]) :
		error += 100.0 * abs(real[i] - predict[i])/real[i]
	error /= real.shape[0]
	return error

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
	result_val =  float(row[header.index("prime_tot_ttc")])
	for f_i, feature in enumerate(row[:33]) :
		if number_info[f_i] > 1 :
			my_dict = feature_info_dict.get(f_i, {})
			test = my_dict.get(feature, None)
			if test is None :
				my_dict[feature] = len(my_dict)
			feature_info_dict[f_i] = my_dict

min_info = np.zeros((33, ), dtype=np.float32)
max_info = np.zeros((33, ), dtype=np.float32)

for row in train_dataset :
	for i in range(33) :
		if number_info[i] == 1 and i is not 6 :
			if row[i] is not "NON" :
				val = float(row[i])
				if i==1 or i==2 : val = 2016-val
				if val < min_info[i] : min_info[i] = val
				if val > max_info[i] : max_info[i] = val

print "info generation done."

train_array = np.zeros((300000, 255))
train_target = np.zeros((300000))
test_array = np.zeros((30000, 255))
test_dataset_ids = []

for index, row in enumerate(train_dataset) :
    train_array[index,:] = feature_generator(row, number_info, min_info, max_info, feature_info_dict)
    train_target[index] = float(row[header.index("prime_tot_ttc")])
for index, row in enumerate(test_dataset):
    test_array[index,:] = feature_generator(row, number_info, min_info, max_info, feature_info_dict)
    test_dataset_ids.append(row[0])

print "start training"

dtrain = xgb.DMatrix(train_array, label=train_target)
#dval = xgb.DMatrix(train_array[240000:], label=train_target[240000:])
dtest = xgb.DMatrix(test_array)

params = {'booster' : 'gbtree', 'objective' : 'reg:linear', 'eval_metric':'rmse', 'silent':1}
#params = {'colsample_bytree': 0.8488490422818511, 'silent': 0, 'eval_metric': 'rmse', 'alpha': 0.15882957654660312, 'subsample': 0.9066084141747773, 'eta': 0.1974169546237201, 'objective': 'reg:linear', 'lambda': 2.0982405114864813, 'max_depth': 10, 'booster': 'gbtree'}
num_rounds = 1000

# try to find best hyperparameters by random search.
min_rmse = 999999.99999
min_params = {}
min_number = 0
result_add = np.zeros((1000, ))
result_count = 0

while True :
	eta = random.random()*0.19+0.01 # 0.01~0.2
	max_depth = random.randint(5, 10) # 5~10
	subsample = random.random()*0.5+0.5 # 0.5~1
	colsample_bytree = random.random()*0.5+0.5 # 0.5~1
	lambda_val = random.random()*2.5 + 0.5 # 0.5~3
	alpha_val = random.random()*0.5 # 0~0.5

	params['eta'] = eta
	params['max_depth'] = max_depth
	params['subsample'] = subsample
	params['colsample_bytree'] = colsample_bytree
	params['lambda_val'] = lambda_val
	params['alpha_val'] = alpha_val

	cvresult = xgb.cv(params, dtrain, 1000, nfold=3)
	result_matrix = pd.Series.as_matrix(cvresult['train-rmse-mean'])
	now_min = np.amin(result_matrix)
	now_argmin = np.argmin(result_matrix)

	if now_min < min_rmse :
		print "New record : ", now_min, "Params : ", params
		min_params = copy.deepcopy(params)
		min_number = now_argmin
		min_rmse = now_min

	result_count += 1
	result_add += result_matrix

	if result_count % 5 == 0 :
		print "Count : ", result_count, min_rmse
		f = open('cv_results_'+str(result_count/5)+'.csv', 'wb')
		bst = xgb.train(min_params, dtrain, 1000)
		last_result = bst.predict(dtest)

		with f as csvfile :
		    fieldnames = ['ID', 'COTIS']
		    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
		    writer.writeheader()
		    for index, value in enumerate(last_result) :
		    	writer.writerow({'ID' : test_dataset_ids[index], 'COTIS' : value})

		f.close()

'''
bst = xgb.train(params, dtrain, num_rounds)
val_result = bst.predict(dval)
mape_value = mape(train_target[240000:], val_result)
print "MAP for validation : ", mape_value

last_result = bst.predict(dtest)

with open('result_xgb_re.csv', 'wb') as csvfile :

    fieldnames = ['ID', 'COTIS']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

    writer.writeheader()
    for index, value in enumerate(last_result) :
    	writer.writerow({'ID' : test_dataset_ids[index], 'COTIS' : value})
'''