import numpy as np

def conv_forward(X, W, pad=0):
    '''
    The forward computation for a convolution function

    Arguments:
    X -- output activations of the previous layer, numpy array of shape
    (n_H_prev, n_W_prev, filters_X) assuming input channels = filters_X
    (e.g. 3 for RGB channels and then kernel/filter depth for the deeper ones)

    W -- Weights, numpy array of size (filters, f, f, filters_prev)
         assuming number of filters = filters

    Returns:
    H -- conv output, numpy array of size (n_H, n_W)
    cache -- cache of values needed for conv_backward() function
    '''
    # Zero-padding
    X_pad = np.pad(X, ((pad, pad), (pad, pad), (0, 0)), 'constant')

    # Retrieving dimensions from X's shape
    (n_H_prev, n_W_prev, filters_X) = X.shape
    # Retrieving dimensions from W's shape
    (filters, f, f, filters_prev) = W.shape
    # print("\nX.shape=%s\nW.shape=%s" % (X.shape, W.shape))

    if(filters_X != filters_prev):
        print("X and W shapes not correct. X.shape: %s, W.shape: %s"
              % (X.shape, W.shape))
    # Compute the output dimensions assuming padding and stride = 1
    n_H = n_H_prev - f + 1 + 2*pad
    n_W = n_W_prev - f + 1 + 2*pad
    # Initialize the output H with zeros
    H = np.zeros((filters, n_H, n_W))
    # Looping over vertical(h) and horizontal(w) axis of output volume
    for h in range(n_H):
        for w in range(n_W):
            x_slice = X_pad[h:h+f, w:w+f]
            # print("x_slice: %s, %s" % (x_slice, x_slice.shape))
            H[:, h, w] = np.sum(x_slice * W, axis=(1, 2, 3))

    # Saving information in 'cache' for backprop
    cache = (X, W)

    return H.transpose(1, 2, 0), cache


#H, cache = conv_forward(Xtr2[4,:,:,1:3], W1[:,:,:,1:3],1)
#print("H.shape=%s, H[:2,:4,:4]:\n %s" % (H.shape, H[:2,:4,:4]))


def conv_backward(dH, cache):
    '''
    The backward computation for a convolution function

    Arguments:
    dH -- gradient of the cost with respect to output of the conv layer (H), 
          numpy array of shape (n_H, n_W, filters) with filters being the
          output kernel size(i.e. number of depth layers/filters in dH)
    cache -- cache of values needed for the conv_backward(), 
             output of conv_forward()

    Returns:
    dX -- gradient of the cost with respect to input of the conv layer (X), 
          numpy array of shape (n_H_prev, n_W_prev, filters_X)
    
    dW -- gradient of the cost with respect to the weights of the 
          conv layer (W), numpy array of shape (filters, f, f, filters_prev)
          assuming number of filters = filters that will end up as layers in dH
          filters_prev are the depth/kernel/filter layers of dX, needs to be
          the same numer as filters_X
    '''

    # Retrieving information from the "cache"
    (X, W) = cache

    # Retrieving dimensions from X's shape
    (n_H_prev, n_W_prev, filters_X) = X.shape

    # Retrieving dimensions from W's shape
    f = W.shape[1]

    # Retrieving dimensions from dH's shape
    (n_H, n_W, filters) = dH.shape

    # Initializing dX, dW with the correct shapes
    dX = np.zeros(X.shape)
    dW = np.zeros(W.shape)

    # Looping over vertical(h) and horizontal(w) axis of the output
    for h in range(n_H):
        for w in range(n_W):
            for depth in range(filters):
                x_slice = W[depth] * dH[h, w, depth]
                dX[h:h+f, w:w+f, :] += x_slice
                dW[depth] += X[h:h+f, w:w+f] * dH[h, w, depth]

    return dX, dW


def printsome(X, W, H):
        print("""

X[0:2, 0:3, 0:3]
%s

W[0:2, 0:2, 0:3, 0:3]
%s

H[0:2, 0:3, 0:3]
%s

""" % (X[0:2, 0:3, 0:3], W[0:2, 0:2, 0:3, 0:3], H[0:2, 0:3, 0:3]
          ))