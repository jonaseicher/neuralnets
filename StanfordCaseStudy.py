# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:27:05 2018

@author: Jonas
"""
import numpy as np
import pylab


def generateSpiralData(N=100, D=2, K=3):
    """
    N = 100  # examples per class
    D = 2  # dimensions
    K = 3  # classes
    """
    X = np.zeros((N*K, D))  # data matrix (1 example per row)
    y = np.zeros(N*K, dtype='uint8')  # class labels

    r = np.linspace(0.0, 1, N)  # vektor(100) ranging 0-1,

    for j in range(K):
        ix = range(N*j, N*(j+1))  # vektor(100) ranging 0-100,100-200,200-300
        t = np.linspace(j*4, (j+1)*4, N) + np.random.rand(N)*0.5
        # t is vektor(100) ranging 0-4, 4-8, 8-12
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y


class StanfordCaseStudy(object):

    def __init__(self, data_dimensions, output_classes):
        self.D = data_dimensions
        self.K = output_classes
        self.W = 0.01 * np.random.rand(self.D, self.K)
        self.b = np.zeros((1, K))
        self.k = 0
        self.j = 0

    def L_softmax(self, X, y, reg=0.01):
        scores = X.dot(self.W) + self.b
        num_examples = X.shape[0]
        exp_scores = np.exp(scores)
        propabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(propabilities[range(num_examples), y])
        data_loss = np.sum(correct_logprobs)/num_examples
        reg_loss = 0.5*reg*np.sum(self.W*self.W)
        loss = data_loss + reg_loss
        self.k += 1
        if self.k%interval == 0:
            print("------------Scores------------")
            print(scores[printset])
            print("------------Exp(Scores)------------")
            print(exp_scores[printset])
            print("------------Sums------------")
            print(np.sum(exp_scores, axis=1, keepdims=True)[printset])
            print("------------Propabilities------------")
            print(propabilities[printset])
            print("------------Sanity check props sum to 1------------")
            print(np.sum(propabilities, axis=1)[printset])
            print("------------log propabilities of correct class (loss)---------")
            print(correct_logprobs[printset])
            print("------------Average data loss over all examples------------")
            print(data_loss)
            print("------------Regulation loss------------")
            print(reg_loss)
        return loss, propabilities

    def SGD_softmax(self, loss, propabilities, reg=0.01):
        num_examples = propabilities.shape[0]
        dscores = np.copy(propabilities)
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples
        
        dW = np.dot(X.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)
        dW += reg*self.W
        self.j += 1
        if self.j%interval == 0:
            print("------------df (Derivative of Scores)------------")
            print(dscores[printset])
            print("------------dW - delta of weights (unregulated)------------")
            print(dW)
            print("------------dW - delta of bias------------")
            print(db)
            print("------------dW - delta of weights with regulation derivative-")
            print(dW)
        return dW, db


N = 100  # examples per class
D = 2  # dimensions
K = 3  # classes
reg = 1e-3  # regulation strength (lambda)
sgd_stepsize = 1e-0

X, y = generateSpiralData(N, D, K)
pylab.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=pylab.cm.Spectral)
pylab.show()

printset = [(99, 102, 210)]
interval = 49

nn = StanfordCaseStudy(D, K)

for i in range(200):
    loss1, props = nn.L_softmax(X, y, reg)
    dW1, db1 = nn.SGD_softmax(loss1, props)
    nn.W += -sgd_stepsize * dW1
    nn.b += -sgd_stepsize * db1
    if i%10 == 0:
        print('iteration %d: loss: %f' % (i, loss1))
