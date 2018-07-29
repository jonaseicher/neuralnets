# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 17:36:10 2018

@author: Jonas
"""
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        return dict


def showImage(X, interpol='spline36'):
    import matplotlib.pyplot as plt
    img = X.reshape(3, 32, 32).transpose([1, 2, 0])
    plt.imshow(img, interpolation=interpol)


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
