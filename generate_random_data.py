#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:10:46 2017

@author: aalen
"""

import scipy.io as sio
import numpy as np
import random
import lmdb
import caffe

# read the dataset
data_mat = sio.loadmat('./data/PaviaU.mat')['paviaU']#[:,:,:100]
data_mat_gt = sio.loadmat('./data/PaviaU_gt.mat')['paviaU_gt']
DB_NAME = 'paviau'

a = np.average(data_mat)
b = np.var(data_mat)**0.5

# generate the training data including the virtual samples
data_mat_dict = {}

for idx_i in xrange(data_mat_gt.shape[0]):
    for idx_j in xrange(data_mat_gt.shape[1]):
        label = data_mat_gt[idx_i, idx_j]
        if not label in data_mat_dict.keys():
            data_mat_dict[label] = []
        if not label == 0:
            spectral = (data_mat[idx_i, idx_j] - a) / b
            data_mat_dict[label].append([spectral, label, 'U', (idx_i,idx_j)])

for key in data_mat_dict.keys():
    class_data = data_mat_dict.get(key)
    random.shuffle(class_data)
    sample_card = 200
    _train = class_data[:sample_card]
    test = class_data[sample_card-10:sample_card]

    train = []
    for item_a in _train:
        for item_b in _train:
            for idx in xrange(2):
                q = random.uniform(-1,2)
                item = [item_a[0]*q+item_b[0]*(1-q)]+item_a[1:]
                train.append(item)

    random.shuffle(train)
    train = _train + train
    data_mat_dict[key] = [train,test]

data_train = []
data_test = []
for item in data_mat_dict:
    if item == 0:
        continue

    temp_data = data_mat_dict[item][0]
    random.shuffle(temp_data)
    temp_data = temp_data[:70000]
    data_train = data_train + temp_data

    temp_data = data_mat_dict[item][1] * 20
    random.shuffle(temp_data)
    temp_data = temp_data[:200]
    data_test = data_test + temp_data

random.shuffle(data_train)
random.shuffle(data_test)

# write the LMDB and record the coordinates

coord_set = set([])
db_train_spect = lmdb.open('%s_spect_lmdb'%(DB_NAME), map_size=1e12)
db_train_label = lmdb.open('%s_label_lmdb'%(DB_NAME), map_size=1e12)
db_test_spect = lmdb.open('%s_spect_test_lmdb'%(DB_NAME), map_size=1e12)
db_test_label = lmdb.open('%s_label_test_lmdb'%(DB_NAME), map_size=1e12)

with db_train_spect.begin(write=True) as txn_train_spect,\
     db_train_label.begin(write=True) as txn_train_label,\
     db_test_spect.begin(write=True) as txn_test_spect,\
     db_test_label.begin(write=True) as txn_test_label,\
     open('%s_coord.txt'%(DB_NAME),'w') as coord_train, \
     open('%s_test_coord.txt'%(DB_NAME),'w') as coord_test:
    for idx, item in enumerate(data_train):

        spectral = item[0]
        label = item[1] - 1
        DB_name = item[2]
        coord = item[3]
        #if idx % 10000 == 0:
        #    print "AD: %4d"%(idx / 10000)
        #lmdb train
        datum = caffe.io.array_to_datum(np.array(spectral)[np.newaxis, np.newaxis, :])
        txn_train_spect.put("{:0>10d}".format(idx), datum.SerializeToString())

        datum = caffe.io.array_to_datum(np.array([label])[np.newaxis, np.newaxis, :])

        txn_train_label.put("{:0>10d}".format(idx), datum.SerializeToString())

        if not str(coord) in coord_set:
            coord_train.write(str(coord)+'\n')
            coord_set.add(str(coord))


    for idx, item in enumerate(data_test):

        spectral = item[0]
        label = item[1] - 1
        DB_name = item[2]
        coord = item[3]

        #lmdb test
        datum = caffe.io.array_to_datum(np.array(spectral)[np.newaxis, np.newaxis, :])
        txn_test_spect.put("{:0>10d}".format(idx), datum.SerializeToString())

        datum = caffe.io.array_to_datum(np.array([label])[np.newaxis, np.newaxis, :])

        txn_test_label.put("{:0>10d}".format(idx), datum.SerializeToString())

        if not str(coord) in coord_set:
            coord_test.write(str(coord)+'\n')
            coord_set.add(str(coord))
db_train_spect.close()
db_train_label.close()
db_test_spect.close()
db_test_label.close()
