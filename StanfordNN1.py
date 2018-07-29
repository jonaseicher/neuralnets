# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:30:43 2018

@author: Jonas
"""

import numpy as np
import pylab
import cifar10 as cf10
import time

Xtr, Ytr, Xte, Yte, label_names, filenames = cf10.loadCIFAR10()


def L(X, y, W):
    t1 = time.time()
    num_train = X.shape[0]  # 50000
    scores = X.dot(W)  # 50000x10 matrix
    margins = []
    loss = []
    yi_scores = scores[np.arange(scores.shape[0]), y]
    # print(scores.shape)
    # print("yi scores shape: %s" % yi_scores.shape)
    margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1)
    margins[np.arange(num_train), y] = 0
    lossPerSample = np.sum(margins, axis=1)
    loss = np.mean(lossPerSample)
    # print ('weight loss: %s' % (0.5 * np.sum(W * W)))
    print('delta t1: %s' % (time.time() - t1))
    t2 = time.time()
    binary = margins
    # print(binary[0])
    binary[margins > 0] = 1
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), y] = -row_sum.T
    # print(binary[0])
    # print(X[0])
    dW = np.dot(X.T, binary)
    dW /= num_train
    # print('W: %s, dW: %s, W.shape: %s, dW.shape: %s'\
     #     % (W[0], dW[0], W.shape, dW.shape))
    print('delta t2: %s' % (time.time() - t2))
    return np.sum(loss), dW

