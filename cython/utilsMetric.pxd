from __future__ import division, print_function
import time, math

import numpy as np 
cimport numpy as np 
from libc.math cimport exp as c_exp
from libc.math cimport log as c_log
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline randomQuery(int n):
    cdef int i, j, k

    i = np.random.randint(n)
    j = np.random.randint(n)
    while (j == i):
        j = np.random.randint(n)
    k = np.random.randint(n)
    while (k == i) | (k == j):
        k = np.random.randint(n)
    q = [i, j, k]
    return q

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double normK(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y, np.ndarray[DTYPE_t, ndim=2] K):
    return np.dot(x, np.dot(K, y))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double tripletScoreK(np.ndarray[DTYPE_t, ndim=2] K, np.ndarray[DTYPE_t, ndim=2] M_t):
    """
    Compute the score of a triplet = <K, M_t> = trace(M_t @ K)
    """
    cdef int p = K.shape[0]
    cdef double score
    cdef int i
    score = 0.

    for i in range(p):
        score = score + np.dot(K[:,i], M_t[:,i])
    return score 
    # return np.trace(np.dot(M_t, K))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double queryScoreK(np.ndarray[DTYPE_t, ndim=2]K, np.ndarray[DTYPE_t, ndim=2]X, list q):
    """
    Given kernel K, feature vectors X, and triplet q=[i,j,k] returns score = 
    ||x_i - x_k||_K^2 - ||x_i - x_j||_k^2, the distance wrt the metric K
    This is equivalent to: score = 2*X[i].T*K*X[j] - 2*X[i].T*K*X[k] - X[j].T*K*X[j] + X[k].T*K*X[k]
    If score > 0 then the kernel agrees with the triplet constraint, otherwise it does not 

    Usage:
    score = tripletScoreK(K,X,[3,4,5])
    """
    cdef int i, j, k
    i, j, k = q[0], q[1], q[2]
    cdef np.ndarray[DTYPE_t, ndim=2] M_t = (np.outer(X[i], X[j]) + np.outer(X[j], X[i]) \
              - np.outer(X[i], X[k]) - np.outer(X[k], X[i])\
              - np.outer(X[j], X[j]) + np.outer(X[k], X[k]))
    return tripletScoreK(K, M_t)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline lossK(np.ndarray[DTYPE_t, ndim=2] K, np.ndarray[DTYPE_t, ndim=3] M):
    """
    Compute empirical and logistic loss from triplets S, 
    with feature vectors X on kernel K
    """
    cdef int num_t = M.shape[0]
    cdef int i
    cdef double emp_loss, log_loss, loss_ijk
    emp_loss = 0.  # 0/1 loss
    log_loss = 0.  # logistic loss

    for i in range(num_t):
        loss_ijk = tripletScoreK(K, M[i])
        if loss_ijk <= 0:
            emp_loss = emp_loss + 1.
        log_loss = log_loss + c_log(1 + c_exp(-loss_ijk))
    return emp_loss / num_t, log_loss / num_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double tripletScoreGradient(np.ndarray[DTYPE_t, ndim=2] XKX, t):
    """
    Compute the score of a triplet = <K, M_t> = trace(M_t @ K)
    """
    i,j,k = t
    return XKX[k,k] - XKX[j,j]-2*(XKX[i,k]- XKX[i,j]) 
    # return np.trace(np.dot(M_t, K))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=2] partialGradientK(np.ndarray[DTYPE_t, ndim=2] K, np.ndarray[DTYPE_t, ndim=2] M_t, np.ndarray[DTYPE_t, ndim=2] XKX, t):
    """
    Compute partial gradient from triplet q, on kernel estimate K 
    with feature vectors X. For triplet i,j,k = q, 
    let score = 2*X[i].T*K*X[j] - 2*X[i].T*K*X[k] - X[j].T*K*X[j] + X[k].T*K*X[k]
    Each triplet t has loss \ell(score_t), where \ell(x) is a convex function, 
    such as the logistic loss in this case.

    Inputs:
    ndarray[float] (pxp) K: p by p kernel matrix
    ndarray[float] (nxp) X: matrix where each row is a feature vector

    Returns:
    ndarray(float) (pxp) G: partial gradient WRT triplet q

    Usage:
    G = partialGradient(K,X,q)
    """
    # cdef double score  = np.trace(np.dot(M_t, K))
    # tripletScoreK(K, M_t)
    return -1. / (1. + c_exp(tripletScoreGradient(XKX, t))) * M_t.T

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=2] fullGradient(np.ndarray[DTYPE_t, ndim=2] K, np.ndarray[DTYPE_t, ndim=3] M, np.ndarray[DTYPE_t, ndim=2] X, S):
    """
    See partial gradient code for specifics. Computes full gradient for set of tripets S.
    The full gradient is equal to the sum of the partials. 

    Inputs:
    ndarray[float] (pxp) K: p by p kernel matrix
    ndarray[float] (nxp) X: matrix where each row is a feature vector
    list[list[int]]      S: list of triplets

    Returns:
    ndarray(float) (pxp) G: full gradient for triplet set S

    Usage:
    G = partialGradient(K,X,q)
    """
    cdef np.ndarray[DTYPE_t, ndim=2] G, XKX
    cdef int num_t = M.shape[0]
    XKX = np.dot(X, np.dot(K, X.T))
    G = np.zeros((K.shape[0], K.shape[1]))
    for i,t in enumerate(S):
        G += partialGradientK(K, M[i], XKX, t)
    return G / num_t


