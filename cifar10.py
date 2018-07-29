# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 17:36:10 2018

@author: Jonas
"""
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        return dict


def showImage(X, interpol='spline36'):
    img = X.reshape(3, 32, 32).transpose([1, 2, 0])
    plt.imshow(img, interpolation=interpol)

def showWeight(W):
    Wimg = np.copy(W) - np.min(W)
    Wimg  /= np.max(Wimg)
    plt.axis('off')
    plt.imshow(Wimg)
    
def plotWeights(W):
    Wimg = np.copy(W) - np.min(W)
    Wimg  /= np.max(Wimg)
    print("Wimg min %s, Wimg max %s, Wimg avg %s" % (np.min(Wimg), np.max(Wimg), np.average(Wimg)))
    print("W min %s, W max %s, W avg %s" % (np.min(W), np.max(W), np.average(W)))
    fig=plt.figure(figsize=(14, 14))
    columns = 8
    rows = 5
    for i in range(1, W.shape[0] + 1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(Wimg[i-1])
    plt.show()
    
    
def plotH(H):
    Himg = np.copy(H.transpose(1,2,0))
    Himg -= np.min(Himg)
    Himg /= np.max(Himg)
    print("Himg min %s, max %s, avg %s" % (np.min(Himg), np.max(Himg), np.average(Himg)))
    print("H min %s, max %s, avg %s" % (np.min(H), np.max(H), np.average(H)))
    fig=plt.figure(figsize=(14, 14))
    columns = 4
    rows = 5
    for i in range(1, Himg.shape[0] + 1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(Himg[i-1])
    plt.show()
    
    
def loadCIFAR10():
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

    #  batch = b1[b'batch_label']
    filenames = b1[b'filenames'] + b2[b'filenames'] + b3[b'filenames'] + \
                b4[b'filenames'] + b5[b'filenames']

    Ytr = b1[b'labels'] + b2[b'labels'] + b3[b'labels'] + b4[b'labels'] + \
          b5[b'labels']
    # 3072 = 1024 x 3(rgb) = 32x32x3 first 1024 red channel, then green, then blue
    Xtr = np.concatenate((b1[b'data'], b2[b'data'], b3[b'data'], b4[b'data'],
                          b5[b'data']))

    Xte = test[b'data']
    Yte = test[b'labels']
    return Xtr, Ytr, Xte, Yte, label_names, filenames
