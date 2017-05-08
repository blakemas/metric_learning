from __future__ import division, print_function
import time, math

import numpy as np 
cimport numpy as np 
from libc.math cimport exp as c_exp
from libc.math cimport log as c_log
cimport cython

cpdef triplets(np.ndarray[DTYPE_t, ndim=2] K, np.ndarray[DTYPE_t, ndim=2] X, int pulls, double steepness=1., noise=False):
    S = []
    cdef int n = X.shape[0]
    cdef list q 
    cdef int t 
    cdef double score 
    for t in range(0, pulls):
        q = randomQuery(n)
        score = queryScoreK(K, X, q)
        if score < 0:
            q = [q[i] for i in [0, 2, 1]]
        if noise:
            if np.random.rand() > 1. / (1. + c_exp(-1. * abs(score))):
                q = [q[i] for i in [0, 2, 1]]
        S.append(q)
    return S

cpdef np.ndarray[DTYPE_t, ndim=3] M_set(list S, np.ndarray[DTYPE_t, ndim=2] X):
    cdef int n, p, i, j, k
    n = X.shape[0]
    p = X.shape[1]
    cdef int num_t = len(S)
    cdef np.ndarray[DTYPE_t, ndim=3] M = np.zeros((num_t, p,p))

    for t in range(num_t):
        i, j, k =  S[t]
        M[t] = (np.outer(X[i], X[j]) + np.outer(X[j], X[i]) \
             -  np.outer(X[i], X[k]) - np.outer(X[k], X[i])\
              - np.outer(X[j], X[j]) + np.outer(X[k], X[k]))
    return M

cpdef np.ndarray[DTYPE_t, ndim=2] features(int n, int p, double scale):
    cdef np.ndarray[DTYPE_t, ndim=2] X = np.random.randn(n, p)/np.sqrt(p)*scale
    return X

cpdef np.ndarray[DTYPE_t, ndim=2] kernel(int p, int d, double scale, sparse):
    cdef np.ndarray[DTYPE_t, ndim=2] subK, K 
    cdef np.ndarray[long, ndim=1] inds 
    cdef int r, i, j, c
    if sparse:
        subK = scale*np.random.randn(d, d)/np.sqrt(d)
        subK = np.dot(subK.T, subK)
        inds = np.arange(p)
        np.random.shuffle(inds)
        inds = inds[:d]             # get d random indicies

        K = np.zeros((p, p))
        K[[[i] for i in inds], inds] = subK
    else:
        K = scale*np.random.randn(p, d)/np.sqrt(d)
        K = np.dot(K, K.T)
    
    return K

def norm_L12(A):
    return np.sum(np.linalg.norm(A, axis=1))

def norm_nuc(A):
    return np.trace(A)

def getScore(K, M_t):
    return tripletScoreK(K, M_t)

def getLoss(K, M):
    return lossK(K, M)


def computeKernel(np.ndarray[DTYPE_t, ndim=2] X, list S, int d, double lam,
                  regularization='L12', 
                  double c1=1e-4, 
                  double rho=0.5, 
                  int maxits=100, 
                  double epsilon=1e-3, 
                  verbose=False,
                  Kstart = None):
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
    cdef int n, p, t, inner_t
    cdef double dif, alpha, emp_loss_0, log_loss_0, emp_loss_k, log_loss_k, normG
    cdef list log_loss, emp_loss
    cdef np.ndarray[DTYPE_t, ndim=2] K, K_old, G
    cdef np.ndarray[DTYPE_t, ndim=3] M = M_set(S, X)
    cdef int bounce = 4
    dif = np.finfo(float).max
    n = X.shape[0]
    p = X.shape[1]
    if Kstart is None:
        K = kernel(p, p, 1, False)
    else:
        K = Kstart
    t = 0
    alpha = 10.
    log_loss = []
    emp_loss = []

    while t < maxits:
        K_old = K
        emp_loss_0, log_loss_0 = lossK(K_old, M)
        t += 1
        G = fullGradient(K_old, M)
        normG = np.linalg.norm(G, ord='fro')                               # compute gradient
        if regularization == 'norm_nuc':
            K = project_nucNorm(K_old - alpha * G, lam)
        elif regularization == 'norm_L12':
            K = alternating_projection(K_old - alpha * G, lam, bounce)

        # stopping criteria
        if dif < epsilon or normG < epsilon*(1+log_loss_0) or alpha < epsilon:
            log_loss.append(log_loss_0)
            emp_loss.append(emp_loss_0)
            print("Exiting at iterate %d because stopping condition satisfied" % t)
            break

        # backtracking line search
        emp_loss_k, log_loss_k = lossK(K, M)
        inner_t = 0         # number of steps back
        while log_loss_k > log_loss_0 - c1 * alpha * normG**2:
            alpha = alpha * rho
            if regularization == 'norm_nuc':
                K = project_nucNorm(K_old - alpha * G, lam)
            elif regularization == 'norm_L12':
                K = alternating_projection(K_old - alpha * G, lam, bounce)
            emp_loss_k, log_loss_k = lossK(K, M)
            inner_t += 1
            if inner_t > 10:
                break
        alpha = 1.1*alpha
        dif = np.abs(log_loss_0 - log_loss_k)
        if verbose:
            print({'iter': t,
                   'emp_loss': emp_loss_k,
                   'log_loss': log_loss_k,
                   'dif': dif,
                   'normG': normG,
                   'back_steps': inner_t,
                   'alpha': alpha})
        log_loss.append(log_loss_k)
        emp_loss.append(emp_loss_k)
    return K, emp_loss, log_loss


def computeKernelEpochSGD(np.ndarray[DTYPE_t, ndim=2] X, list S, int d, double lam,
                          regularization='L12', 
                          double a = 1,
                          double c1=1e-5, 
                          double rho=0.5, 
                          int maxits_sgd=100,
                          int maxits_gd=30,
                          double epsilon=1e-3, 
                          verbose=False):
    
    cdef double score, outer_loss
    cdef int m = len(S)
    cdef int epoch_length = len(S)    
    cdef int t = 0
    cdef int t_e = 0
    cdef double rel_max_grad = float('inf')
    cdef int p = X.shape[1]
    cdef int n = X.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] K, G
    cdef int bounce = 10
    K = kernel(p, p, 1, False)
    cdef np.ndarray[DTYPE_t, ndim=3] M = M_set(S, X)
    
    while t < maxits_sgd:
        t += 1
        t_e += 1
        # check epoch conditions, udpate step size
        if t_e % epoch_length == 0:
            a = a*(1+a)**-1
            epoch_length = 2*epoch_length
            t_e = 0
            if regularization == 'norm_nuc':
                K = project_nucNorm(K, lam)
            elif regularization == 'norm_L12':
                K = alternating_projection(K, lam, bounce)

            if epsilon>0 or verbose:
                # get losses
                emp_loss, log_loss = lossK(K, M)
                # get gradient and check stopping-time statistics
                G = fullGradient(K, M)
                normG = np.linalg.norm(G, ord='fro')
                print({'iter':t,
                       'epoch':t_e,
                       'emp_loss':emp_loss,
                       'log_loss':log_loss,
                       'G_norm':normG,
                       'alpha':a})
                if rel_max_grad < epsilon:
                    break
                            
        K = K - a*partialGradientK(K, M[np.random.randint(m)])
    return computeKernel(X, S, d, lam,
                         regularization, 
                         c1, 
                         rho, 
                         maxits_gd, 
                         epsilon, 
                         verbose,
                         Kstart = K)

def euclidean_proj_l1ball(v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    cdef int n = v.shape[0]  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    cdef np.ndarray u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    cdef np.ndarray w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w

def euclidean_proj_simplex(v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    cdef int n = v.shape[0]  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

cpdef inline np.ndarray[DTYPE_t, ndim=2] projectPSD(np.ndarray[DTYPE_t, ndim=2] K):
    '''
    Project onto rank d psd matrices
    '''
    cdef np.ndarray[DTYPE_t, ndim=2] V
    cdef np.ndarray[DTYPE_t, ndim=1] D
    D, V = np.linalg.eigh(K)
    D = np.maximum(D, 0)
    return np.dot(np.dot(V, np.diag(D)), V.T)

cpdef inline np.ndarray[DTYPE_t, ndim=2] project_L12(np.ndarray[DTYPE_t, ndim=2] M, double tau):
    """
    Project onto the L12 ball of radius tau
    """
    if np.sum(np.linalg.norm(M, axis=1)) <= tau:
        return M
    cdef np.ndarray[DTYPE_t, ndim=1] row_l2_norms = np.sqrt(np.sum(np.abs(M)**2, axis=1))
    cdef np.ndarray[DTYPE_t, ndim=1] w = euclidean_proj_l1ball(row_l2_norms, tau)
    for i in range(M.shape[0]):
        if row_l2_norms[i] != 0:
            M[i,:] = M[i,:]/row_l2_norms[i]*w[i] 
    return M

cpdef inline np.ndarray[DTYPE_t, ndim=2] alternating_projection(np.ndarray[DTYPE_t, ndim=2] K, double tau, bounce):
    """
    Project onto the L12 ball of radius tau intersect the PSD cone via alternating projection
    """
    for _ in range(bounce):
        K = projectPSD(project_L12(K, tau))
    return K

cpdef project_nucNorm(M, R):
    '''
    Project onto psd nuclear norm ball of radius R
    '''
    cdef int n = M.shape[0]
    if R!=None:
        D, V = np.linalg.eigh(M)
        D = euclidean_proj_simplex(D, s=R)
        M = np.dot(np.dot(V,np.diag(D)),V.transpose());
    return M


