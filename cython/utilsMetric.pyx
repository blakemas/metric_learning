from __future__ import division, print_function
import time, math

import numpy as np 
cimport numpy as np 
from libc.math cimport exp as c_exp
from libc.math cimport log as c_log

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t
cimport cython
import blackbox

cpdef triplets(np.ndarray[DTYPE_t, ndim=2] K, np.ndarray[DTYPE_t, ndim=2] X, int pulls, double steepness=1., noise=False):
    """
    Generate a random set of #pulls triplets
    """
    S = []
    cdef int n = X.shape[0]
    cdef list q 
    cdef int t 
    cdef double score 
    for t in range(0, pulls):
        # get random triplet
        q = randomQuery(n)
        score = queryScoreK(K, X, q)
        # align it so it agrees with Ktrue: "q[0] is more similar to q[1] than
        # q[2]"
        if score < 0:
            q = [q[i] for i in [0, 2, 1]]
        # add some noise
        if noise:
            if np.random.rand() > 1. / (1. + c_exp(-1. * abs(score))):
                #print(1. / (1. + c_exp(-1. * abs(score))))
                q = [q[i] for i in [0, 2, 1]]
        S.append(q)
    return S

cpdef np.ndarray[DTYPE_t, ndim=3] M_set(list S, np.ndarray[DTYPE_t, ndim=2] X):
    """
    Precompute M_t matrices for speed (Note: uses tons of memory)
    """
    cdef int n, p, i, j, k
    n = X.shape[0]
    p = X.shape[1]
    cdef int num_t = len(S)
    cdef np.ndarray[DTYPE_t, ndim=3] M = np.zeros((num_t, p,p))

    for t in range(num_t):
        i, j, k =  S[t]
        # M[t] = (2. * np.outer(X[i], X[j]) - 2. * np.outer(X[i], X[k]) \
        #       - np.outer(X[j], X[j]) + np.outer(X[k], X[k]))
        M[t] = (np.outer(X[i], X[j]) + np.outer(X[j], X[i]) \
             -  np.outer(X[i], X[k]) - np.outer(X[k], X[i])\
              - np.outer(X[j], X[j]) + np.outer(X[k], X[k]))
    return M

cpdef np.ndarray[DTYPE_t, ndim=2] features(int n, int p):
    """
    Generate a set of 0 mean, unit norm feature vectors for the points

    Inputs:
    (int) n: the number of points
    (int) p: the number of dimensions

    Returns:
    ndarray(n x p) X: matrix where each row is a feature vector
    """
    cdef np.ndarray[DTYPE_t, ndim=2] X = np.random.randn(n, p)/np.sqrt(p)
    # X = X/np.linalg.norm(X, axis=1)[:, np.newaxis]
    return X

cpdef np.ndarray[DTYPE_t, ndim=2] kernel(int p, int d):
    """
    Create a p by p symmetric PSD kernel with d^2 non-zero entries

    Inputs:
    (int) p: the total number of features in the feature vectors x
    (int) d: the number of relevant features

    Returns:
    ndarray[float] K: p by p kernel matrix

    Usage: K = kernel(100,5)
    """
    cdef np.ndarray[DTYPE_t, ndim=2] subK, K 
    cdef np.ndarray[long, ndim=1] inds 
    cdef int r, i, j, c 
    # CASE OF GOOD NUCLEAR NORM
    subK = np.sqrt(p/np.sqrt(d))*np.random.randn(d, d)/np.sqrt(d)
    #
    subK = np.dot(subK.T, subK)
    #subK = 0.5 * subK + 0.5 * subK.T        # psd, symmetric submatrix
    inds = np.arange(p)
    np.random.shuffle(inds)
    inds = inds[:d]             # get d random indicies

    K = np.zeros((p, p))
    K[[[i] for i in inds], inds] = subK
    
    # for r, i in enumerate(inds):
    #     for c, j in enumerate(inds):
    #         K[i, j] = subK[r, c]


    #subK = np.sqrt(np.sqrt(d))*np.random.randn(p, d)/np.sqrt(d)
    #K = np.dot(subK, subK.T)
    
    return K

def norm_L12(A):
    return np.sum(np.linalg.norm(A, axis=1))

def getGrad(K, M):
    return fullGradient(K, M)

def getScore(K, M_t):
    return tripletScoreK(K, M_t)

def getLoss(K, M):
    return lossK(K, M)

def getPartial(K, M_t):
    return partialGradientK(K, M_t)

def L12_projection(K, tau):
    return project_L12(K, tau)

@blackbox.record
def computeKernel(np.ndarray[DTYPE_t, ndim=2] X, list S, int d, double lam,
                                                            regularization='L12', 
                                                            double c1=1e-5, 
                                                            double rho=0.5, 
                                                            int maxits=100, 
                                                            double epsilon=1e-3, 
                                                            verbose=False):
    """
    Compute a sparse, symmetric PSD kernel from triplet observations S feature vectors X,
    using projected gradient descent. Specifically, we wish to solve the following optimation:
    K = \arg\min_{K PSD} \sum_{t \in S} \log(1+exp(-score_t)) + \lambda*||K||_1
    where the 'score' of a triplet t = ||x_i - x_k||_K^2 - ||x_i - x_j||_k^2, the distance wrt the kernel K.
    This is solved via projected gradient descent.

    Inputs:
    ndarray[float] (nxp)        X: matrix where each row is a feature vector
    list[list[int]]             S: list of triplets
    [int]                       d: the number of relevant features (d <= p)
    [float]                   lam: the regularization parameter for the L1 loss, lambda
    [int]                  maxits: the maximum number of iterations
    [float]               epsilon: the stopping condition
    [boolean]             verbose: Controls verbosity

    Returns: 
    ndarray[float] (pxp)        K: estimated sparse, low rank Kernel matrix
    list[float]          emp_loss: empirical loss at each iteration
    list[float]          log_loss: logistic loss at each iteration
    """
    # select proximal operator
    if not (regularization == "L12" or regularization == "nucNorm" 
            or regularization == "L1" or regularization == "alternating"): 
        raise AssertionError("Please choose 'alternating', 'L1', L12 or 'nucNorm' for parameter 'regularization' ")

    cdef int n, p, t, inner_t, bounce, parity
    cdef double dif, alpha, emp_loss_0, log_loss_0, emp_loss_k, log_loss_k, normG
    cdef list log_loss            # logistic loss
    cdef list emp_loss            # empirical loss
    cdef np.ndarray[DTYPE_t, ndim=2] K, K_old, G
    cdef np.ndarray[DTYPE_t, ndim=3] M = M_set(S, X)

    if regularization == 'alternating':
        bounce = 4

    dif = np.finfo(float).max       # realmax 
    n = X.shape[0]
    p = X.shape[1]
    K = kernel(p, p)            # get a random dense Kernel to initialize
    t = 0                   # iteration count
    alpha = 200.             # step size
    log_loss = []
    emp_loss = []

    while t < maxits:
        K_old = K
        emp_loss_0, log_loss_0 = lossK(K_old, M)
        t += 1
        alpha = 1.3 * alpha                                 # update step size
        G = fullGradient(K_old, M)
        normG = np.linalg.norm(G, ord='fro')                               # compute gradient
        if regularization == "L12":
            K = prox_L12(K_old - alpha * G, lam, d)                                    
        elif regularization == "L1":
            K = prox_L1(K_old - alpha * G, lam, d)
        elif regularization == "nucNorm":
            K = prox_nucNorm(K_old - alpha * G, lam, d)
        elif regularization == 'alternating':
            K = alternating_projection(K_old - alpha * G, lam, bounce)

        # stopping criteria
        if dif < epsilon or normG < epsilon or alpha < epsilon:
            log_loss.append(log_loss_0)
            emp_loss.append(emp_loss_0)
            print("Exiting at iterate %d because stopping condition satisfied" % t)
            break

        # backtracking line search
        emp_loss_k, log_loss_k = lossK(K, M)
        inner_t = 0         # number of steps back
        while log_loss_k > log_loss_0 - c1 * alpha * normG**2:
            alpha = alpha * rho
            if regularization == "L12":
                K = prox_L12(K_old - alpha * G, lam, d)
            elif regularization == "L1":
                K = prox_L1(K_old - alpha * G, lam, d)
            elif regularization == "nucNorm":
                K = prox_nucNorm(K_old - alpha * G, lam, d)
            elif regularization == 'alternating':
                K = alternating_projection(K_old - alpha * G, lam, bounce)
            emp_loss_k, log_loss_k = lossK(K, M)
            inner_t += 1
            if inner_t > 10:
                break
        alpha = 1.1*alpha
        dif = np.linalg.norm(K - K_old, ord='fro')
        
        blackbox.logdict({'iter': t,
                        'emp_loss': emp_loss_k,
                        'log_loss': log_loss_k,
                        'dif': dif,
                        'back_steps': inner_t,
                        'alpha': alpha})
        blackbox.save(verbose=verbose)

        log_loss.append(log_loss_k)
        emp_loss.append(emp_loss_k)
    return K, emp_loss, log_loss


