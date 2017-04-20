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
    """
    Outputs a triplet [i,j,k] chosen uniformly at random from all possible triplets 
    and score = abs( ||x_i - x_k||^2 - ||x_i - x_j||^2 )

    Inputs:
            (numpy.ndarray) X : matrix from which n is extracted from and score is derived

    Outputs:
        [(int) i, (int) j, (int) k] q : where k in [n], i in [n]-k, j in [n]-k-j
        (float) score : signed distance to current solution (positive if it agrees, negative otherwise)

    Usage:
            q,score = getRandomQuery(X)
    """
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
    """
    Weighted inner product with respect to PSD matrix K
    """
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
    # return 2.*normK(X[i], X[j], K) - 2.*normK(X[i], X[k], K) - normK(X[j],
    # X[j], K) + normK(X[k], X[k], K)
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
cdef inline np.ndarray[DTYPE_t, ndim=2] groupLasso(np.ndarray[DTYPE_t, ndim=2] K, double lam):
    """
    Group lasso prox operator with groups = rows (note: K is symmetric)
    """
    cdef int i
    cdef double nrm
    cdef int p = K.shape[0]

    for i in range(p):
        nrm = np.linalg.norm(K[i])
        K[i] = K[i] * (1. - lam/nrm) * (nrm > lam)
    return K

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=1]softThreshold(np.ndarray[DTYPE_t, ndim=1] z, double lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=2] projectPSD(np.ndarray[DTYPE_t, ndim=2] K):
    '''
    Project onto rank d psd matrices
    '''
    cdef np.ndarray[DTYPE_t, ndim=2] V
    cdef np.ndarray[DTYPE_t, ndim=1] D
    D, V = np.linalg.eigh(K)
    D = np.maximum(D, 0)
    return np.dot(np.dot(V, np.diag(D)), V.T)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=2] projectPSDRankD(np.ndarray[DTYPE_t, ndim=2] K, int d):
    '''
    Project onto rank d psd matrices
    '''
    cdef np.ndarray[DTYPE_t, ndim=2] V
    cdef np.ndarray[DTYPE_t, ndim=1] D 
    cdef int n, i

    n = K.shape[0]
    D, V = np.linalg.eigh(K)
    perm = D.argsort()
    bound = np.max(D[perm][-d], 0)
    
    for i in range(n):
        if D[i] < bound:
            D[i] = 0
    return np.dot(np.dot(V, np.diag(D)), V.transpose())

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=2] prox_L12(np.ndarray[DTYPE_t, ndim=2] K, double lam, int d):
    return projectPSD(groupLasso(K, lam))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=1] project_L1(np.ndarray[DTYPE_t, ndim=1] v, double tau):
    """
    Project onto the L1 ball of radius tau
    """
    if np.linalg.norm(v, ord=1) <= tau:
        return v
    cdef np.ndarray[DTYPE_t, ndim=1] u = np.sort(v)[::-1]
    cdef np.ndarray[DTYPE_t, ndim=1] sv = np.cumsum(u)
    cdef int rho = int(np.nonzero(u > np.divide((sv - tau), 1 + np.arange(len(u))))[0][-1])
    cdef double theta = np.maximum(0., (sv[rho] - tau)/(rho + 1.))
    return np.multiply(np.sign(v), np.maximum(np.abs(v) - theta, 0.))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=2] project_L12(np.ndarray[DTYPE_t, ndim=2] M, double tau):
    """
    Project onto the L12 ball of radius tau
    """
    cdef np.ndarray[DTYPE_t, ndim=1] row_l2_norms = np.sqrt(np.sum(np.abs(M)**2, axis=1))
    cdef np.ndarray[DTYPE_t, ndim=1] w = project_L1(row_l2_norms, tau)
    return M/row_l2_norms[:, np.newaxis] * w[:, np.newaxis]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=2] alternating_projection(np.ndarray[DTYPE_t, ndim=2] K, double tau, parity):
    """
    Project onto the L12 ball of radius tau
    """
    if parity:
        return projectPSD(K)
    else:
        return project_L12(K, tau)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=2] prox_L1(np.ndarray[DTYPE_t, ndim=2] K, double lam, int d):
    K = np.sign(K) * np.maximum(np.abs(K) - lam, 0)         # soft threshold (note: function only takes vectors)
    return projectPSDRankD(K, d)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=2] prox_nucNorm(np.ndarray[DTYPE_t, ndim=2] K, double lam, int d): 
    """
    Nuclear norm regularization and then project onto rank D PSD
    """
    cdef np.ndarray[DTYPE_t, ndim=2] V
    cdef np.ndarray[DTYPE_t, ndim=1] D
    cdef int n, i 

    n = K.shape[0]
    D, V = np.linalg.eigh(K)
    D = np.maximum(D - lam, 0)        # soft threshold and project onto PSD
    return np.dot(np.dot(V, np.diag(D)), V.transpose())



@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=2] partialGradientK(
                            np.ndarray[DTYPE_t, ndim=2] K, np.ndarray[DTYPE_t, ndim=2] M_t):
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
    return -1. / (1. + c_exp(tripletScoreK(K, M_t))) * M_t.T

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=2] fullGradient(np.ndarray[DTYPE_t, ndim=2] K, np.ndarray[DTYPE_t, ndim=3] M):
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
    cdef np.ndarray[DTYPE_t, ndim=2] G
    cdef int num_t = M.shape[0]
    G = np.zeros((K.shape[0], K.shape[1]))
    for t in range(num_t):
        G += partialGradientK(K, M[t])
    return G / num_t








