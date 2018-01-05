import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    Inputs have dimension D, there are C classes, and we operate on minibatches of N examples.
    Inputs:
    W: A numpy array of shape (D, C) containing weights
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,y[i]] -= X[i]
                dW[:,j] += X[i]
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W ** 2)
    dW += reg * W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    W = W.astype(np.float64)
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    # W: (3073, 10)
    # X: (5000, 3073)
    num_train = X.shape[0]
    scores = X.dot(W) # (5000,10)
    yi_scores = scores[np.arange(num_train), y] #(5000, 1)
    margins = np.maximum(0, scores - yi_scores[:, np.newaxis] + 1.0) #(5000, 10)
    margins[np.arange(num_train), y] = 0
    loss = np.sum(margins)
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    
    counts = (margins > 0).astype(int)
    counts[range(num_train), y] -= np.sum(counts, axis = 1)
    
    dW = X.T.dot(counts)
    dW /= num_train
    dW += reg * W
    return loss, dW
