# -*- coding: utf-8 -*-
"""
Created on Sat Jul 8 17:36:10 2018

@author: Jonas
"""
import numpy as np
import cifar10 as cf10

num_test = 3

Xtr, Ytr, Xte, Yte, label_names, filenames = cf10.loadCIFAR10()

def L_i(x, y, W):
    delta = 1.0
    scores = W.dot(x)
    correct_class_score = scores[y]
    D = W.shape[0]  # number of classes, eg 10
    loss_i = 0.0
    for j in range(D):
        if j == y:
            continue
        loss_i += max(0, scores[j] - correct_class_score + delta)
    #print(loss_i)
    return loss_i


def L_i_vectorized(x, y, W):
    delta = 1.0
    scores = W.dot(x)
    margins = np.maximum(0, scores - scores[y] + delta)
    margins[y] = 0
    print(margins)
    loss_i = np.sum(margins)
    #print(loss_i)
    return loss_i


def L(X, Y, W):
    delta = 1.0
    scores = X.dot(W.transpose())  # 50000x10 matrix
    margins = []
    loss = []
    for i in range(scores.shape[0]):
        margin = np.maximum(0, scores[i] - scores[i][Y[i]] + delta)
        margin[Y[i]] = 0
        margins.append(margin)
        loss.append(np.sum(margin))
    return loss
 

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

L(Xtr,Ytr,W)
    