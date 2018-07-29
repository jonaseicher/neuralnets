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


# Conv plus ReLu
for sample in range(samples):
    X = Xtr2[sample]
    H1, cache1 = conv.conv_forward(X, W1, pad)
    # print(H1[0:2, 0:3, 0:3])
    # ReLu: https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
    H1_relu = np.copy(H1)
    H1_relu[H1 < 0] = 0
    # print(H1_relu.shape)
    for m in range(15):
        for n in range(15):
            # pool
            x_slice = H1_relu[2*m:2*m+2, 2*n:2*n+2]
            P1[m, n] = np.max(x_slice, axis=(0, 1))

    # Conv
    H2, cache2 = conv.conv_forward(P1, W2, pad)
    H2_relu = np.copy(H2)
    H2_relu[H2 < 0] = 0

#    for kernel in range(depth1):
#        print('H1_relu %s %s' % (kernel, np.sum(H1_relu[kernel])))
#    for kernel2 in range(depth2):
#        print('H2_relu %s %s' % (kernel2, np.sum(H2_relu[kernel2])))

    x = H2.flatten()
    scores = x.dot(W3)
    # softmax
    ex = np.exp(scores)
    probs[sample] = ex/np.sum(ex, keepdims=True)
    correct_logprobs = -np.log(probs[sample, Ytr[0]])
    data_loss += correct_logprobs
    dscores = np.copy(probs)
    dscores[sample, Ytr[sample]] -= 1

    # backprop
    dW3 = np.dot(H2_relu.reshape(3042, 1), dscores[sample].reshape(1, 10))
    print("np.sum(np.abs(W3)): %s" % np.sum(np.abs(W3)))
    print("np.sum(np.abs(dW3)): %s" % np.sum(np.abs(dW3)))

    dH2 = np.dot(dscores[sample], W3.T).reshape(13, 13, depth2)
    dH2[H2 <= 0] = 0
    print("np.sum(np.abs(H2)): %s" % np.sum(np.abs(H2)))
    print("np.sum(np.abs(dH2)): %s" % np.sum(np.abs(dH2)))

    # Conv Backprop from dH2
    dP1, dW2 = conv.conv_backward(dH2, cache2)
    
    print("\nnp.sum(np.abs(W2)): %s" % np.sum(np.abs(W2)))
    print("np.sum(np.abs(dW2)): %s" % np.sum(np.abs(dW2)))
    print("np.sum(np.abs(P1)): %s" % np.sum(np.abs(P1)))
    print("np.sum(np.abs(dP1)): %s" % np.sum(np.abs(dP1)))

    # reverse pooling
    dH1 = np.zeros(H1.shape)
    for m in range(15):
        for n in range(15):
            dH1[2*m:2*m+2, 2*n:2*n+2] = dP1[m, n]

    # ReLu backprop
    dH1[H1 <= 0] = 0
    # Conv Backprop from dH1 to input data
    dX, dW1 = conv.conv_backward(dH1, cache1)

    print("\nnp.sum(np.abs(W1)): %s" % np.sum(np.abs(W1)))
    print("np.sum(np.abs(dW1)): %s" % np.sum(np.abs(dW1)))
    print("np.sum(np.abs(X)): %s" % np.sum(np.abs(X)))
    print("np.sum(np.abs(dX)): %s" % np.sum(np.abs(dX)))

data_loss /= samples
loss_over_time.append(data_loss)
print("Data Loss: %s" % data_loss)
dscores = np.copy(probs)
dscores[range(samples), Ytr[:samples]] -= 1
print("dscores: %s" % np.round(dscores, 3))

# backpropate the gradient

# backprop to conv layer 2

# dW3 = np.dot(conv_layer1)
# todo: read about convnet backprop


plt.ylabel('loss')
plt.grid(True)
plt.plot(np.arange(len(loss_over_time)), loss_over_time, '-',
         label='data_loss')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()










    # z[z<0] = 0  # most performant ReLu: https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy

## some hyperparameters
#step_size = 1e-0
#reg = 1e-7  # regularization strength
#
## gradient descent loop

#for i in range(10000):
#
#    # evaluate class scores, [N x K]
#    hidden_layer = np.maximum(0, np.dot(X, W) + b)  # ReLU
#    scores = np.dot(hidden_layer, W2) + b2
#
#    # compute the class probabilities
#    exp_scores = np.exp(scores)
#    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]
#    # compute the loss: average cross-entropy loss and regularization
#    correct_logprobs = -np.log(probs[range(num_examples), y])
#    data_loss = np.sum(correct_logprobs)/num_examples
#    reg_loss = 0.5*reg*np.sum(W*W)
#    loss = data_loss + reg_loss
#    loss_over_time.append(data_loss)
#    reg_loss_over_time.append(reg_loss)
#
#    # compute the gradient on scores
#    dscores = np.copy(probs)
#    dscores[range(num_examples), y] -= 1
#    dscores /= num_examples
#    # backpropate the gradient to the parameters (W2,b2)
#    dW2 = np.dot(hidden_layer.T, dscores)
#    db2 = np.sum(dscores, axis=0, keepdims=True)
#    # calculate gradient of hidden layer (backprop)
#    dhidden = np.dot(dscores, W2.T)
#    # now backprop ReLU
#    dhidden[hidden_layer <= 0] = 0
#    # backprop gradient to first layer
#    dW = np.dot(X.T, dhidden)
#    db = np.sum(dhidden, axis=0, keepdims=True)
#
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