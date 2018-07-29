
def conv_forward(X, W):
    '''
    The forward computation for a convolution function

    Arguments:
    X -- output activations of the previous layer, numpy array of shape 
         (n_H_prev, n_W_prev) assuming input channels = 1
    W -- Weights, numpy array of size (f, f) assuming number of filters = 1

    Returns:
    H -- conv output, numpy array of size (n_H, n_W)
    cache -- cache of values needed for conv_backward() function
    '''

    # Retrieving dimensions from X's shape
    (n_H_prev, n_W_prev) = X.shape

    # Retrieving dimensions from W's shape
    (f, f) = W.shape

    # Compute the output dimensions assuming no padding and stride = 1
    n_H = n_H_prev - f + 1
    n_W = n_W_prev - f + 1

    # Initialize the output H with zeros
    H = np.zeros((n_H, n_W))

    # Looping over vertical(h) and horizontal(w) axis of output volume
    for h in range(n_H):
        for w in range(n_W):
            x_slice = X[h:h+f, w:w+f]
            H[h,w] = np.sum(x_slice * W)
    
    # Saving information in 'cache' for backprop
    cache = (X, W)

    return H, cache
