# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 19:56:21 2018

@author: Jonas
"""
import cifar10
import numpy as np
import matplotlib.pylab as plt

loss_over_time = []
reg_loss_over_time = []

Xtr, Ytr, Xte, Yte, label_names, filenames = cifar10.loadCIFAR10()

# auf -1 bis 1 normieren
Xtr2 = Xtr/128-1
depth = 12
W1 = 1e-1 * np.random.randn(depth, 3, 3, 3)
depth2 = 18
W2 = 1e-2 * np.random.randn(depth2, depth, 3, 3)
outputs = 10
# reshape to pictures
Xtr2 = Xtr2.reshape(Xtr2.shape[0], 3, 32, 32).transpose([0, 2, 3, 1])
# to show: plt.imshow(Xtr2[0]+0.5)  it works!

conv_layer1 = np.zeros((depth, 30, 30))
b1 = np.zeros(conv_layer1.shape)
pool_layer1 = np.zeros((depth, 15, 15))
conv_layer2 = np.zeros((depth2, 13, 13))
c2_neurons = depth2 * 13 * 13
W3 = 1e-2 * np.random.randn(c2_neurons, outputs)
b2 = np.zeros(conv_layer2.shape)
b3 = np.zeros(outputs)
samples = 2
data_loss = 0
probs = np.zeros((samples, outputs))
# Conv plus ReLu
for sample in range(samples):
    for kernel in range(depth):
        for m in range(30):
            for n in range(30):
                x3 = 0
                for rgb in range(3):
                    x = Xtr2[sample, m:m+3, n:n+3, rgb]
                    mat = x * W1[kernel]  # Conv
                    s = np.sum(mat)
                    x2 = max(0, s)  # ReLu
                    x3 += x2
                conv_layer1[kernel, m, n] = x3 + b1[kernel, m, n]

            for m in range(15):
                for n in range(15):
                    # pool
                    pool_layer1[kernel, m, n] = np.max(conv_layer1[kernel, 2*m:2*m+2, 2*n:2*n+2])

    for kernel2 in range(depth2):
        for kernel1 in range(depth):
            for m in range(13):
                for n in range(13):
                    x = pool_layer1[kernel1, m:m+3, n:n+3]
                    # Conv und ReLu
                    x2 = max(0, np.sum(x * W2[kernel2]))
                    conv_layer2[kernel2, m, n] = x2 + b2[kernel2, m, n]

    x = conv_layer2.flatten()
    scores = x.dot(W3) + b3
    # softmax
    ex = np.exp(scores)
    probs[sample] = ex/np.sum(ex, keepdims=True)
    correct_logprobs = -np.log(probs[sample, Ytr[0]])
    data_loss += correct_logprobs
    
data_loss /= samples
loss_over_time.append(data_loss)
print(data_loss)
dscores = np.copy(probs)
dscores[range(samples), Ytr[:samples]] -= 1
print(dscores)

# backpropate the gradient to the parameters (W2,b2)
db3 = np.sum(dscores, axis=0, keepdims=True)
dW3 = np.dot(conv_layer2.reshape(3042,1), dscores[sample].reshape(1,10))

# backprop to conv layer 2
dconv2 = np.dot(dscores, W3.T).reshape(samples, depth2, 13, 13)

dconv2[:, conv_layer2 < 0] = 0   # must use conv-layer2 before relu
db2 = np.sum(dconv2, axis=0, keepdims=True)
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