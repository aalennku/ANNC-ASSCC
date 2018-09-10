import caffe
import numpy as np


class CenterLossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    feature_len = 32
    num_class = 20
    center = np.zeros((num_class, feature_len, 1, 1))
    counter = 0

    def setup(self, bottom, top):
        # check input pair
        self.alpha = 0.5
        if len(bottom) != 2:
            raise Exception("Need two inputs as features and labels.")
        self.batch_size = bottom[0].data.shape[0]

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].data.shape[0] != bottom[1].data.shape[0]:
            raise Exception("Inputs must have the same channel numbers.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.diff_loss = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        center_update = np.zeros((self.num_class, self.feature_len, 1, 1))
        self.center = np.zeros((self.num_class, self.feature_len, 1, 1))
        features = bottom[0].data
        labels = bottom[1].data
        center_update_counter = [0.1]*self.num_class
        for i in xrange(labels.shape[0]):
            label = int(labels[i])
            center_update[label] = center_update[label] + features[i]
            if center_update_counter[label] == 0.1:
                center_update_counter[label] = 1
            else:
                center_update_counter[label] += 1

        center_update = np.array(
            [item / float(cnt) for item,
             cnt in zip(center_update, center_update_counter)])
        if self.counter == 0:
            self.center = center_update
        else:
            self.center = self.center + (
                center_update - self.center) * self.alpha

        loss = 0
        # for i in xrange(labels.shape[0]):
        #    self.diff[i] = features[i] - self.center[int(labels[i])]
        # print self.diff[i].reshape(100)[:3]
        # print features[i].reshape(100)[:3]
        # print self.center[int(labels[i])].reshape(100)[:3]
        # print '========================'
        for i in xrange(labels.shape[0]):
            label = int(labels[i])
            self.diff[i] = np.zeros_like(features[i]).astype(np.float)
            nearest_idx = 0
            nearest_dist = 999999999
            if label < 9:
                label_range = (0, 9)
            else:
                label_range = (9,18)
            for idx_j in xrange(*label_range):
                if idx_j == label:
                    self.diff[i] += features[i] - self.center[idx_j]
                    self.diff_loss[i] = features[i] - self.center[idx_j]
                else:
                    continue
                    self.diff[i] += (self.center[idx_j] - features[i]) * 0.05
                    #dist = np.sum((self.diff[i] - self.center[idx_j])**2)
                    #if dist < nearest_dist:
                    #    nearest_dist = dist
                    #    nearest_idx = idx_j
            #self.diff[i] += (self.center[nearest_idx] - features[i]) * 0.3
        if self.counter % 1000 == 0:
            loss_1 = 0
            loss_2 = 0
            cnt_1, cnt_2 = 1, 1#not zero for not to divide zero
            for idx, item in enumerate(labels):
                if item > 8:
                    loss_2 += np.sum(self.diff_loss[idx]**2)
                    cnt_2 += 1
                else:
                    loss_1 += np.sum(self.diff_loss[idx]**2)
                    cnt_1 += 1
            print "======================Info: loss_1: %.4f loss_2: %.4f." % (loss_1/cnt_1, loss_2/cnt_2)

        loss = np.sum(self.diff_loss**2)
        loss = loss / labels.shape[0]
        top[0].data[...] = loss

        if self.counter % (100) == 0:
            max = 0
            min = 1000
            # for i in xrange(self.center.shape[0]):
            #    for j in xrange(self.center.shape[0]):
            for i in xrange(9):
                for j in xrange(9):
                    if i == j:
                        continue
                    dist = np.sum((self.center[i] - self.center[j])**2)
                    if dist > max:
                        max = dist
                    if dist < min and dist != 0:
                        min = dist

            if self.counter % (1000) == 0:
                print "======================Info-0: max: %.4f min: %.4f." % (max, min)
            with open('center_dist.txt', 'a') as f:
                f.write('%d\t%f\t%f\n'%(self.counter,max,min))
            max = 0
            min = 1000
            for i in xrange(9, 18):
                for j in xrange(9, 18):
                    if i == j:
                        continue
                    dist = np.sum((self.center[i] - self.center[j])**2)
                    if dist > max:
                        max = dist
                    if dist < min and dist != 0:
                        min = dist
            if self.counter % (1000) == 0:
                print "======================Info-1: max: %.4f min: %.4f." % (max, min)
        self.counter += 1

    def backward(self, top, propagate_down, bottom):
        # for i in range(2):
        #    if not propagate_down[i]:
        #        continue
        #    if i == 0:
        #        sign = 1
        #    else:
        #        sign = -1
        bottom[0].diff[...] = self.diff / bottom[0].num  # not clear
