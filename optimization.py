# -*- coding: utf-8 -*-
"""
Created on Sat Jul 8 17:36:10 2018

@author: Jonas
"""
import numpy as np
import pylab
import cifar10 as cf10
import time

Xtr, Ytr, Xte, Yte, label_names, filenames = cf10.loadCIFAR10()


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


def Lopt(X, Y, W):
    t1 = time.time()
    delta = 1.0
    num_train = X.shape[0]  # 50000
    scores = X.dot(W)  # 50000x10 matrix
    margins = []
    loss = []
    yi_scores = scores[np.arange(scores.shape[0]), Y]
    # print(scores.shape)
    # print("yi scores shape: %s" % yi_scores.shape)
    margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1)
    margins[np.arange(num_train), Y] = 0
    lossPerSample = np.sum(margins, axis=1)
    loss = np.mean(lossPerSample)
    # print ('weight loss: %s' % (0.5 * np.sum(W * W)))
    print('delta t1: %s' % (time.time() - t1))
    t2 = time.time()
    binary = margins
    # print(binary[0])
    binary[margins > 0] = 1
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), Y] = -row_sum.T
    # print(binary[0])
    # print(X[0])
    dW = np.dot(X.T, binary)
    dW /= num_train
    # print('W: %s, dW: %s, W.shape: %s, dW.shape: %s'\
     #     % (W[0], dW[0], W.shape, dW.shape))
    print('delta t2: %s' % (time.time() - t2))
    return np.sum(loss), dW


lossOverTime = []
correctness = []

def randomSearch(i=10):
    bestloss = float("inf")
    for num in range(i):
        W = np.random.randn(10,3072) * 0.0003
        loss, dW = Lopt(Xtr, Ytr, W)
        if (loss < bestloss):
            bestloss = loss
            bestW = W
        print('in attempt %d the loss was %f, best %f' % (num, loss, bestloss))
        lossOverTime.append(loss)
    return bestW, bestloss

def localSearch(i=10):
    bestloss = float("inf")
    W = np.random.randn(10, 3072) * 0.003
    Wstep = np.random.randn(10,3072) * 0.0003
    for num in range(i):
        Wtry = W + Wstep
        loss, dW = Lopt(Xtr, Ytr, Wtry)
        if loss < bestloss:
            W = Wtry
            bestloss = loss
            Wstep *= 1.2
        else:
            Wstep = np.random.randn(10, 3072) * 0.0003
        print('in attempt %d the loss was %f, best %f' % (num, loss, bestloss))
        yte_predict = np.argmax(Xte.dot(W.T), axis=1)
        print('correctness: %s' % np.mean(yte_predict == Yte))
        lossOverTime.append(bestloss)
    return W, bestloss

def gradientSearch(i=10):
    bestloss = float("inf")
    W = np.random.randn(3072, 10) * 0.003
    for num in range(i):
        loss, dW = Lopt(Xtr, Ytr, W)
        sumW = np.sum(np.abs(W))
        sumdW = np.sum(np.abs(dW))
        dW *= 0.005*sumW/sumdW
        #print('W: %s, dW: %s, dW/W = %s' % (sumW, sumdW, sumdW/sumW))
        W -= dW
        print('in attempt %d the loss was %f, best %f' % (num, loss, bestloss))
        yte_predict = np.argmax(Xte.dot(W), axis=1)
        print('correctness: %s' % np.mean(yte_predict == Yte))
        bestloss = loss
        lossOverTime.append(bestloss)
        correctness.append(np.mean(yte_predict == Yte))
    return W.T, bestloss
# bestW, bestloss = randomSearch(10)
# bestW, bestloss = localSearch(1)
bestW, bestloss = gradientSearch(100)
#import cProfile
#print(cProfile.run('gradientSearch()'))

scores = Xte.dot(bestW.transpose())  # 10000x10
Yte_predict = np.argmax(scores, axis=1)  # 10000 class numbers (0-9)
print(np.mean(Yte_predict == Yte))

pylab.plot(lossOverTime)
#pylab.plot(correctness)