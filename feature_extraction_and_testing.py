#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:30:00 2017

@author: aalen
"""

import caffe
import numpy as np
import scipy.io as sio
import sys

K = 9
DB_NAME = 'paviau'
# feature extraction
model_weights = './train/models/paviaU_iter_80000.caffemodel'
model_def = './train/deploy.prototxt'
caffe.set_mode_gpu()
net = caffe.Net(model_def, model_weights, caffe.TEST)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))

data_mat = sio.loadmat('./data/PaviaU.mat')['paviaU']
data_mat_gt = sio.loadmat('./data/PaviaU_gt.mat')['paviaU_gt']

a = np.average(data_mat)
b = np.var(data_mat)**0.5

data_normalized = np.zeros_like(data_mat).astype(np.float)
for idx_i in xrange(data_mat.shape[0]):
    for idx_j in xrange(data_mat.shape[1]):
        data_normalized[idx_i, idx_j] = \
        (data_mat[idx_i, idx_j] - a)/b

data = transformer.preprocess('data', data_normalized)
net.blobs['data'].data[0] = data
ans = net.forward()['prob']
feature = net.blobs['conv3'].data

# generate the class centers
# generate a mask to exclude the training data
mask = np.ones_like(data_mat_gt)
with open('paviau_coord.txt','r') as tr:
    train_list = tr.readlines()

train_data_list = {}
train_set = set()
for item in train_list:
    idx_i, idx_j = eval(item)
    mask[idx_i,idx_j] = 0
    label = data_mat_gt[idx_i,idx_j] - 1
    if not label in train_data_list:
        train = []
        train_data_list[label] = train
    if not item in train_set:
        train_data_list[label].append(data_normalized[idx_i,idx_j])
        train_set.add(item)

def in_train_data(idx_i,idx_j):
    if str((idx_i,idx_j))+'\n' in train_set :
        return True
    else:
        return False

model_def_1 = './train/deploy_1.prototxt'
net_1 = caffe.Net(model_def_1, model_weights, caffe.TEST)
transformer = caffe.io.Transformer({'data':net_1.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
feature_len = 32

center_list = []
for label in xrange(9):
    cnt = 0
    center = np.zeros(feature_len)
    for item in train_data_list[label]:
        data = transformer.preprocess('data', item.reshape((1,1,103)))
        net_1.blobs['data'].data[0] = data
        net_1.forward()
        center += net_1.blobs['conv3'].data.reshape(feature_len)
    center = center / len(train_data_list[label])
    center_list.append(center)

#evaluate
def progress_bar(idx, total_number):
    percent = idx*100./total_number
    sys.stdout.write('\r['+'#'*(int(percent/2))+'='*(50-int(percent/2))+'] %4.1f%%'%(percent))
    sys.stdout.flush()

correct = 0
fail = 0
kernel_list = xrange(2,10)
stat_data = dict()
predict_dict = [0] * (K + 1)
groundt_dict = [0] * (K + 1)
shape = data_mat_gt.shape

for idx_i in xrange(shape[0]):
    progress_bar(idx_i, shape[0]-1)
    for idx_j in xrange(shape[1]):
        if data_mat_gt[idx_i,idx_j] == 0:
            continue
        if in_train_data(idx_i,idx_j):
            continue

        predict_data = []

        for kernel in kernel_list:
            av_feature = \
                np.average(feature[0,:,max(idx_i-kernel+1,0):min(idx_i+kernel,shape[0]),\
                   max(idx_j-kernel+1,0):min(idx_j+kernel,shape[1])].reshape((32,-1)),axis=1,\
                   weights=mask[max(idx_i-kernel+1,0):min(idx_i+kernel,shape[0]),\
                   max(idx_j-kernel+1,0):min(idx_j+kernel,shape[1])].reshape(-1))
            dist = 77777777777 # a big number
            label_av = -1
            for idx, center in enumerate(center_list):
                new_dist = np.sum((av_feature - center)**2)#/np.sum((center)**2)
                if dist > new_dist:
                    dist = new_dist
                    label_av = idx
            label_av += 1
            predict_data.append([dist,label_av,kernel])

        predict_label = np.array([item[1] for item in predict_data])
        weights = np.array([1./item[0] for item in predict_data])
        vote = [np.sum((predict_label==i)*weights) for i in xrange(0,17)]

        label = np.argmax(vote)

        predict_dict[label] += 1
        groundt_dict[data_mat_gt[idx_i,idx_j]] += 1

        if not data_mat_gt[idx_i,idx_j] in stat_data:
            stat_data[data_mat_gt[idx_i,idx_j]] = [0,0]
        if label == data_mat_gt[idx_i,idx_j]:
            correct += 1
            stat_data[data_mat_gt[idx_i,idx_j]][0] += 1
            stat_data[data_mat_gt[idx_i,idx_j]][1] += 1
        else:
            fail += 1
            stat_data[data_mat_gt[idx_i,idx_j]][1] += 1
sys.stdout.write('\n')
sum_correct = 0
for key in stat_data:
    print '%2d, %5d, %5d, %.4f'%(key,stat_data[key][0],\
                                stat_data[key][1],\
                                stat_data[key][0]*1./stat_data[key][1])
    sum_correct += stat_data[key][0]*1./stat_data[key][1]
print "%6d/%6d"%(correct,fail)
oa = correct/(correct+fail*1.)
aa = sum_correct/K
pe = np.sum(np.array(predict_dict)*np.array(groundt_dict))*1./(np.sum(np.array(predict_dict))**2)
kc = (oa-pe)/(1-pe)
print 'overall accuracy: %.4f'%(oa)
print 'average accuracy: %.4f'%(aa)
print 'kappa coefficien: %.4f'%(kc)
