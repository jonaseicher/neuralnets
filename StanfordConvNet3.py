# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 19:56:21 2018

@author: Jonas
"""
import cifar10
import numpy as np
import matplotlib.pylab as plt
import methods_conv1 as conv


loss_over_time = []
reg_loss_over_time = []

Xtr, Ytr, Xte, Yte, label_names, filenames = cifar10.loadCIFAR10()

# auf -1 bis 1 normieren
Xtr2 = Xtr/128-1

# reshape to pictures
Xtr2 = Xtr2.reshape(Xtr2.shape[0], 3, 32, 32).transpose([0, 2, 3, 1])
# to show: plt.imshow(Xtr2[0]+0.5)  it works!

depth0 = Xtr2.shape[3]
depth1 = 12
depth2 = 18
outputs = 10

learning_rate = 0.01
pad = 0
width0 = 32
width1 = width0 - 2 + 2*pad
width1_2 = int(width1/2)  # pooling
width2 = width1_2 - 2 + 2*pad
c2_neurons = depth2 * width2 * width2

# Weights
W1 = 1e-1 * np.random.randn(depth1, 3, 3, depth0)  # conv1
W2 = 1e-1 * np.random.randn(depth2, 3, 3, depth1)  # conv2
W3 = 1e-2 * np.random.randn(c2_neurons, outputs)  # fc

P1 = np.zeros((15, 15, depth1))  # pool layer 1

samples = 2
data_loss = 0
probs = np.zeros((samples, outputs))


def train(X, W1, W2, W3, pad=0):

    # Conv 1
    H1, cache1 = conv.conv_forward(X, W1, pad)

    # ReLu 1
    H1_relu = np.copy(H1)
    H1_relu[H1 < 0] = 0

    # Pool
    for m in range(15):
        for n in range(15):
            x_slice = H1_relu[2*m:2*m+2, 2*n:2*n+2]
            P1[m, n] = np.max(x_slice, axis=(0, 1))

    # Conv 2
    H2, cache2 = conv.conv_forward(P1, W2, pad)

    # ReLu 2
    H2_relu = np.copy(H2)
    H2_relu[H2 < 0] = 0

    # FC 1
    x = H2_relu.flatten()
    scores = x.dot(W3)

    # Softmax
    ex = np.exp(scores)
    probs[sample] = ex/np.sum(ex, keepdims=True)
    print("Propability of correct class: %s" % probs[sample, Ytr[sample]])
    loss = -np.log(probs[sample, Ytr[sample]])
    dscores = np.copy(probs)
    dscores[sample, Ytr[sample]] -= 1

    # Backprop FC 1
    dW3 = np.dot(H2_relu.reshape(3042, 1), dscores[sample].reshape(1, 10))
    dH2 = np.dot(dscores[sample], W3.T).reshape(13, 13, depth2)

    # Backprop ReLu 2
    dH2[H2 <= 0] = 0

    # Backprop Conv 2
    dP1, dW2 = conv.conv_backward(dH2, cache2)

    # Backprop Pool
    dH1 = np.zeros(H1.shape)
    for m in range(15):
        for n in range(15):
            dH1[2*m:2*m+2, 2*n:2*n+2] = dP1[m, n]

    # Backprop ReLu 1
    dH1[H1 <= 0] = 0

    # Backprop Conv 1
    dX, dW1 = conv.conv_backward(dH1, cache1)

    return loss, dW1, dW2, dW3

for epoch in range(15):
    print("np.sum(np.abs(W1)): %s" % np.sum(np.abs(W1)))
    print("np.sum(np.abs(W2)): %s" % np.sum(np.abs(W2)))
    print("np.sum(np.abs(W3)): %s" % np.sum(np.abs(W3)))
    data_loss = 0
    for sample in range(samples):
        loss, dW1, dW2, dW3 = train(Xtr2[sample], W1, W2, W3, pad)
        print("\nnp.sum(np.abs(dW1)): %s" % np.sum(np.abs(dW1)))
        print("np.sum(np.abs(dW2)): %s" % np.sum(np.abs(dW2)))
        print("np.sum(np.abs(dW3)): %s" % np.sum(np.abs(dW3)))
        data_loss += loss
        W1 -= dW1 * learning_rate
        W2 -= dW2 * learning_rate
        W3 -= dW3 * learning_rate

    data_loss /= samples
    loss_over_time.append(data_loss)
    print("\nData Loss: %s" % data_loss)
    dscores = np.copy(probs)
    dscores[range(samples), Ytr[:samples]] -= 1
    print("dscores:\n%s" % np.round(dscores, 3))
    if(epoch > 7 & epoch % 5 == 0):
        learning_rate *= 0.5


plt.ylabel('loss')
plt.grid(True)
plt.plot(np.arange(len(loss_over_time)), loss_over_time, '-',
         label='data_loss')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()




#    dW += reg*W  # regularization gradient
#    dW2 += reg*W2
#    # perform a parameter update
#    W += -step_size * dW
#    b += -step_size * db
#    W2 += -step_size * dW2
#    b2 += -step_size * db2
#    if i % 100 == 0:
#        print("iteration %d: loss %f" % (i, loss))
#        predicted_class = np.argmax(scores, axis=1)
#        print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
#
#
#plt.figure(1)
#plt.subplot(211)
#plt.ylabel('loss')
#plt.grid(True)
#plt.plot(np.arange(len(loss_over_time)), loss_over_time, '-',
#         label='data_loss')
#plt.legend(loc='upper right')
#plt.subplot(212)
#plt.plot(reg_loss_over_time, ':k', label='reg_loss')
#plt.legend(loc='lower right')
#plt.xlabel('epoch')
#plt.ylabel('loss')
#plt.show()
#
#
## test data
#xdim = 32
#ydim = 32
#grid = np.zeros((32*32, 2))  # data matrix (1 example per row)
#x = np.linspace(0.0, 1, 32)  # vektor(100) ranging 0-1,
#y = np.linspace(0.0, 1, 32)
##for j in range(100):
##    ix = range(N*j, N*(j+1))  # vektor(100) ranging 0-100,100-200,200-300
##    t = np.linspace(j*4, (j+1)*4, N) + np.random.rand(N)*0.5
##    # t is vektor(100) ranging 0-4, 4-8, 8-12
##    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
##    y[ix] = j
##return X, y
#
#
## predict
#hidden_layer = np.maximum(0, np.dot(grid, W) + b)  # ReLU
#scores = np.dot(hidden_layer, W2) + b2
#
## compute the class probabilities
#exp_scores = np.exp(scores)
#probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
## plotting: ^(triangles),-(line) --(dashed), g(green), b(blue), k(black),
## o(circles)