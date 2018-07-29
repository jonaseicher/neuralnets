# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 00:47:35 2018

@author: Jonas
"""
# Train a Linear Classifier

import numpy as np
import matplotlib.pyplot as plt


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


N = 100  # examples per class
D = 2  # dimensions
K = 3  # classes
h = 100  # size of hidden layer


X, y = generateSpiralData(N, D, K)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

# initialize parameters randomly
W = 0.01 * np.random.randn(D, h)
b = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

# some hyperparameters
step_size = 1e-0
reg = 1e-7  # regularization strength

printset = [(99, 102, 210)]

# gradient descent loop
num_examples = X.shape[0]
loss_over_time = []
reg_loss_over_time = []
for i in range(10000):

    # evaluate class scores, [N x K]
    hidden_layer = np.maximum(0, np.dot(X, W) + b)  # ReLU
    scores = np.dot(hidden_layer, W2) + b2

    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]
    # compute the loss: average cross-entropy loss and regularization
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss
    loss_over_time.append(data_loss)
    reg_loss_over_time.append(reg_loss)

    # compute the gradient on scores
    dscores = np.copy(probs)
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples
    # backpropate the gradient to the parameters (W2,b2)
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # calculate gradient of hidden layer (backprop)
    dhidden = np.dot(dscores, W2.T)
    # now backprop ReLU
    dhidden[hidden_layer <= 0] = 0
    # backprop gradient to first layer
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)

    dW += reg*W  # regularization gradient
    dW2 += reg*W2
    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2
    if i % 100 == 0:
        print("iteration %d: loss %f" % (i, loss))
        predicted_class = np.argmax(scores, axis=1)
        print('training accuracy: %.2f' % (np.mean(predicted_class == y)))


plt.figure(1)
plt.subplot(211)
plt.ylabel('loss')
plt.grid(True)
plt.plot(np.arange(len(loss_over_time)), loss_over_time, '-',
         label='data_loss')
plt.legend(loc='upper right')
plt.subplot(212)
plt.plot(reg_loss_over_time, ':k', label='reg_loss')
plt.legend(loc='lower right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# test data
xdim = 32
ydim = 32
grid = np.zeros((32*32, 2))  # data matrix (1 example per row)
x = np.linspace(0.0, 1, 32)  # vektor(100) ranging 0-1,
y = np.linspace(0.0, 1, 32)
#for j in range(100):
#    ix = range(N*j, N*(j+1))  # vektor(100) ranging 0-100,100-200,200-300
#    t = np.linspace(j*4, (j+1)*4, N) + np.random.rand(N)*0.5
#    # t is vektor(100) ranging 0-4, 4-8, 8-12
#    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
#    y[ix] = j
#return X, y


# predict
hidden_layer = np.maximum(0, np.dot(grid, W) + b)  # ReLU
scores = np.dot(hidden_layer, W2) + b2

# compute the class probabilities
exp_scores = np.exp(scores)
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
# plotting: ^(triangles),-(line) --(dashed), g(green), b(blue), k(black),
# o(circles)
