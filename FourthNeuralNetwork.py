# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 14:07:26 2018

@author: Jonas
"""

import numpy as np

caterpillars = 10
ladybirds = 10
X = np.array([[0, 1], [1, 1], [1, 2], [2, 3], [3, 5], [5, 8], [8, 13],
              [13, 21], [21, 34], [34, 55], [55, 89], [89, 144],
              [144, 233], [2584, 4181]])

y = np.array([[1], [2], [3], [5], [8], [13], [21], [34], [55],
              [89], [144], [233], [377], [6765]])

testX = np.array([[233, 377], [610, 987]])
testy = np.array([[610], [1597]])
# Normalize

X = X/np.amax(X, axis=0)
y = y/6765

testX = testX/np.amax(testX, axis=0)
testy = testy/6765


class NeuralNetwork(object):
    def __init__(self, Lambda=0):
        # Number of input, hidden layer and output neurons
        self.inputs = 2
        self.outputs = 1
        self.neurons = 3

        # Weights (parameters)
        self.W1 = np.random.randn(self.inputs, self.neurons)
        self.W2 = np.random.randn(self.neurons, self.outputs)

        # Regularization parameter to avoid overfitting
        self.Lambda = Lambda

    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        self.yFat = yHat*6765
        return yHat

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # Derivative of sigmoid function = (1-sigmoid(x))
    def sigmoidPrime(self, x):
        return np.exp(-x)/(1+np.exp(-x))**2

    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        J = 0.5*np.sum(((y-self.yHat)/y)**2)/X.shape[0] +\
            (self.Lambda/2)*(np.sum(self.W1**2) + np.sum(self.W1**2))
        return J

    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3) + self.Lambda*self.W2

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2) + self.Lambda*self.W1

        return dJdW1, dJdW2

    def computeGradient(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    # Helper Functions for interacting with other classes:
    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.neurons * self.inputs
        self.W1 = np.reshape(params[W1_start:W1_end],
                             (self.inputs, self.neurons))
        W2_end = W1_end + self.neurons*self.outputs
        self.W2 = np.reshape(params[W1_end:W2_end],
                             (self.neurons, self.outputs))


def computeNumericalGradient(N, X, y):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for p in range(len(paramsInitial)):
        perturb[p] = e
        N.setParams(paramsInitial + perturb)
        loss2 = N.costFunction(X, y)

        N.setParams(paramsInitial - perturb)
        loss1 = N.costFunction(X, y)

        numgrad[p] = (loss2-loss1)/(2*e)

        perturb[p] = 0

    N.setParams(paramsInitial)

    return numgrad

# BFGS optimization
from scipy import optimize


class trainer(object):
    def __init__(self, N):
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        self.testJ.append(self.N.costFunction(self.testX, self.testy))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = computeNumericalGradient(self.N, X, y)
        return cost, grad

    def train(self, X, y, testX, testy):
        self.X = X
        self.y = y

        self.testX = testX
        self.testy = testy

        self.J = []
        self.testJ = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True,
                                 method='BFGS', args=(X, y), options=options,
                                 callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res


nn = NeuralNetwork(0.000001)
import pylab
grad = nn.computeGradient(X, y)
numgrad = computeNumericalGradient(nn, X, y)
T = trainer(nn)
T.train(X, y, testX, testy)
pylab.plot(T.J)
pylab.plot(T.testJ)
