# author iizukak, 2011
# author cam davidson-pilon, 2012

import numpy as np
import pdb
def gs_cofficient(v1, v2):
    return np.dot(v2, v1) / np.dot(v1, v1)

def multiply(cofficient, v):
    return map((lambda x : x * cofficient), v)

def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)

def gs(X):
    """
    performs the Gramm-Shmidt process to orthonormalize a a matrix of vectors.
    X: vectors to orthonormalize are rows.
    Returns Y, same shape as X, and with orthonormal rows.
    """
    Y = np.zeros_like(X)
    for i in range(len(X)):
        temp_vec = X[i]
        for j in range(i) :
            proj_vec = proj(Y[j,:], X[i])
            temp_vec = temp_vec - proj_vec
        Y[i,:] = temp_vec/np.sqrt( np.dot( temp_vec,temp_vec ) )
    return Y


