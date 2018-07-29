# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 17:36:10 2018

@author: Jonas
"""
import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        return dict


workspace = 'C:/Users/Jonas/.spyder-py3/workspace_jonas'
cifar_dir = '/cifar-10-batches-py'
path = workspace + cifar_dir

b1 = unpickle(path + '/data_batch_1')
b2 = unpickle(path + '/data_batch_2')
b3 = unpickle(path + '/data_batch_3')
b4 = unpickle(path + '/data_batch_4')
b5 = unpickle(path + '/data_batch_5')
test = unpickle(path + '/test_batch')
label_names = unpickle(path + '/batches.meta')[b'label_names']

batch = b1[b'batch_label']
filenames = b1[b'filenames'] + b2[b'filenames'] + b3[b'filenames'] + b4[b'filenames'] + b5[b'filenames']

Ytr = b1[b'labels'] + b2[b'labels'] + b3[b'labels'] + b4[b'labels'] + b5[b'labels']
Xtr = np.concatenate((b1[b'data'], b2[b'data'], b3[b'data'], b4[b'data'], b5[b'data'])) # 3072 = 1024 x 3(rgb) = 32x32x3 first 1024 red channel, then green, then blue
Xte = test[b'data']
Yte = test[b'labels']

num_test = 20

class NearestNeighbor(object):
  def __init__(self, k=1):
    self.k = k

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    # num_test = 50 #X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype='int8')

    # loop over all test rows
    for i in range(num_test):
        # find the nearest training image to the i'th test image
        # using the L1 distance (sum of absolute value differences)
        distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
        sortedDistances = np.sort(distances)
        nearestNeigbourIndexes = []
        predictions = []
        kLabels = []
        for j in range(self.k):
            newIndex = np.where(distances==sortedDistances[j])[0][0]
            nearestNeigbourIndexes.append(newIndex)
            prediction = self.ytr[newIndex]
            predictions.append(prediction)
            kLabel = label_names[prediction]
            kLabels.append(kLabel)

        mostVotes = np.bincount(predictions).argmax()
        Ypred[i] = mostVotes  # predict the label of the nearest example

        print("labelt: %s, testpic: %s" % (label_names[Yte[i]], test[b'filenames'][i]))
        print(kLabels)
        print(label_names[mostVotes])
        print("--------------------------")
      

    return Ypred


nn = NearestNeighbor(7)
nn.train(Xtr,Ytr)
Yte_predict = nn.predict(Xte)

hits = 0
for i in range(num_test):
    if Yte_predict[i] == Yte[i]:
        hits += 1

print( 'probability: %d%%' % ( hits*100/num_test) )
    