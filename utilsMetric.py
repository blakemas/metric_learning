from __future__ import division, print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def triplets(K, X, pulls, steepness=1, noise=False):
    """
    Generate a random set of #pulls triplets
    """
    S = []
    n = X.shape[0]
    for i in range(0, pulls):
        # get random triplet
        q = randomQuery(n)
        score = tripletScoreK(K, X, q)
        # align it so it agrees with Ktrue: "q[0] is more similar to q[1] than
        # q[2]"
        if score < 0:
            q = [q[i] for i in [0, 2, 1]]
        # add some noise
        if noise:
            if np.random.rand() > 1. / (1. + np.exp(-1. * steepness * abs(score))):
                q = [q[i] for i in [0, 2, 1]]
        S.append(q)
    return S

def M_set(S, X):
    n, p = X.shape
    num_t = len(S)
    M = []
    for q in S:
        i, j, k =  q
        M_t = 2. * np.outer(X[i], X[j]) - 2. * np.outer(X[i], X[k]) \
              - np.outer(X[j], X[j]) + np.outer(X[k], X[k])
        M.append(M_t)
    return M


def randomQuery(n):
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
    i = np.random.randint(n)
    j = np.random.randint(n)
    while (j == i):
        j = np.random.randint(n)
    k = np.random.randint(n)
    while (k == i) | (k == j):
        k = np.random.randint(n)
    q = [i, j, k]
    return q


def features(n, p):
    """
    Generate a set of 0 mean, unit variance feature vectors for the points

    Inputs:
    (int) n: the number of points
    (int) p: the number of dimensions

    Returns:
    ndarray(n x p) X: matrix where each row is a feature vector
    """
    X = np.random.rand(n, p)
    X = (X - np.mean(X)) / np.std(X)		# make 0 mean, unit variance
    return X


def kernel(p, d):
    """
    Create a p by p symmetric PSD kernel with d^2 non-zero entries

    Inputs:
    (int) p: the total number of features in the feature vectors x
    (int) d: the number of relevant features

    Returns:
    ndarray[float] K: p by p kernel matrix

    Usage: K = kernel(100,5)
    """
    subK = np.random.rand(d, d)
    subK = np.dot(subK.T, subK)
    subK = 0.5 * subK + 0.5 * subK.T 		# psd, symmetric submatrix
    inds = np.arange(p)
    np.random.shuffle(inds)
    inds = inds[:d]				# get d random indicies

    K = np.zeros((p, p))
    for r, i in enumerate(inds):
        for c, j in enumerate(inds):
            K[i, j] = subK[r, c]
    return projected(K, d)


def normK(x, y, K):
    """
    Weighted inner product with respect to PSD matrix K
    """
    return np.dot(x, np.dot(K, y))


def tripletScoreK(K, X, q):
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
    return np.trace(np.dot(computeMt(X, q), K))


def computeMt(X, q):
    """
    Compute a matrix Mt so that the triplet loss can be written as an inner product
    """
    i, j, k = q
    return (2. * np.outer(X[i], X[j]) - 2. * np.outer(X[i], X[k])
            - np.outer(X[j], X[j]) + np.outer(X[k], X[k]))


def lossK(K, X, S):
    """
    Compute empirical and logistic loss from triplets S, 
    with feature vectors X on kernel K
    """
    emp_loss = 0  # 0/1 loss
    log_loss = 0  # logistic loss
    for q in S:
        loss_ijk = tripletScoreK(K, X, q)
        if loss_ijk <= 0:
            emp_loss = emp_loss + 1.
        log_loss = log_loss + np.log(1 + np.exp(-loss_ijk))
    return emp_loss / len(S), log_loss / len(S)


def softThreshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)


def projected(M, d):
    '''
    Project onto rank d psd matrices
    '''
    n, n = M.shape
    D, V = np.linalg.eigh(M)
    perm = D.argsort()
    bound = np.max(D[perm][-d], 0)
    for i in range(n):
        if D[i] < bound:
            D[i] = 0
    M = np.dot(np.dot(V, np.diag(D)), V.transpose())
    return M


def partialGradientK(K, X, q):
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
    Mt = computeMt(X, q)
    score = tripletScoreK(K, X, q)
    return -1. / (1 + np.exp(score)) * Mt.T


def fullGradient(K, X, S):
    """
    See partial gradient code for specifics. Computes full gradient for set of tripets S.
    The full gradient is equal to the sum of the partials. 

    Inputs:
    ndarray[float] (pxp) K: p by p kernel matrix
    ndarray[float] (nxp) X: matrix where each row is a feature vector
    list[list[int]] 	 S: list of triplets

    Returns:
    ndarray(float) (pxp) G: full gradient for triplet set S

    Usage:
    G = partialGradient(K,X,q)
    """
    G = np.zeros(K.shape)
    for q in S:
        G += partialGradientK(K, X, q)
    return G / len(S)


def proxK(Z, lam, d):
    """
    Composition of projection and soft thresholding prox operators
    """
    p = Z.shape[0]
    Z = softThreshold(Z[:], lam).reshape((p, p))			# soft threshold
    return projected(Z, d)								# project onto PSD cone
    # Z = projected(Z, d)
    # return softThreshold(Z[:],lam).reshape((p,p))


def computeKernel(X, S, d, lam, c1=1e-5, rho=0.5, maxits=100, epsilon=1e-3, verbose=False):
    """
    Compute a sparse, symmetric PSD kernel from triplet observations S feature vectors X,
    using projected gradient descent. Specifically, we wish to solve the following optimation:
    K = \arg\min_{K PSD} \sum_{t \in S} \log(1+exp(-score_t)) + \lambda*||K||_1
    where the 'score' of a triplet t = ||x_i - x_k||_K^2 - ||x_i - x_j||_k^2, the distance wrt the kernel K.
    This is solved via projected gradient descent.

    Inputs:
    ndarray[float] (nxp)   		X: matrix where each row is a feature vector
    list[list[int]] 	   	 	S: list of triplets
    [int] 				    	d: the number of relevant features (d <= p)
    [float]   	   	  	      lam: the regularization parameter for the L1 loss, lambda
    [int] 	  	  		   maxits: the maximum number of iterations
    [float]	 	          epsilon: the stopping condition
    [boolean]	          verbose: Controls verbosity

    Returns: 
    ndarray[float] (pxp)	    K: estimated sparse, low rank Kernel matrix
    list[float]	  		 emp_loss: empirical loss at each iteration
    list[float]	         log_loss: logistic loss at each iteration

    """
    dif = float('inf')
    n, p = X.shape
    K = kernel(p, d)			# get a random Kernel to initialize
    # K = projected(np.random.rand(p,p), d)
    t = 0					# iteration count
    alpha = 200				# step size
    log_loss = []			# logistic loss
    emp_loss = []			# empirical loss

    while t < maxits:
        K_old = K
        emp_loss_0, log_loss_0 = lossK(K_old, X, S)
        t += 1
        alpha = 1.3 * alpha									# update step size
        G = fullGradient(K_old, X, S)								# compute gradient
        K = proxK(K_old - alpha * G, lam, d) 									# take a step
        normG = np.linalg.norm(G, ord='fro')

        # stopping criteria
        if dif < epsilon or normG < epsilon:
            log_loss.append(log_loss_0)
            emp_loss.append(emp_loss_0)
            print("Exiting at iterate %d because stopping condition satisfied" % t)
            break

        # backtracking line search
        emp_loss_k, log_loss_k = lossK(K, X, S)
        inner_t = 0			# number of steps back
        while log_loss_k > log_loss_0 - c1 * alpha * normG**2:
            alpha = alpha * rho
            K = proxK(K_old - alpha * G, lam, d)
            emp_loss_k, log_loss_k = lossK(K, X, S)
            inner_t += 1
            if inner_t > 10:
                break
        alpha = 1.1*alpha

        dif = np.linalg.norm(K - K_old, ord='fro')
        if verbose:
            print("Iterate=%d, emp_loss=%f, log_loss=%f, dif=%f, back_steps=%d, alpha=%f"
                  % (t, emp_loss_k, log_loss_k, dif, inner_t, alpha))
        log_loss.append(log_loss_k)
        emp_loss.append(emp_loss_k)
    return K, emp_loss, log_loss


def alternatingMin(X, S, r, d, c1=1e-5, rho=0.5, maxits=200, epsilon=1e-3, verbose=False):
    """
    Code to do alternating minimzation to find rank d PSD K within L1 ball of radius r
    """
    dif = float('inf')
    n, p = X.shape
    K = kernel(p, d)			# get a random Kernel to initialize
    t = 0
    alpha = 3
    normG = float('inf')

    while t < maxits:
        t += 1
        K_old = K
        alpha = 1.3*alpha
        emp_loss_0, log_loss_0 = lossK(K_old,X,S)

        # stopping criteria
        if normG < epsilon:
            print("Exiting at iterate %d because stopping condition satisfied" % t)
            break

        G = fullGradient(K_old, X, S)
        normG = np.linalg.norm(G, ord='fro')            # norm of gradient
        K = K_old - alpha * G                               # take a step
        emp_loss_k, log_loss_k = lossK(K, X, S)         # compute losses before line search

        if t % 2 == 0:
            if np.linalg.norm(K[:], ord=1) > r:
                K *= r / np.linalg.norm(K[:], ord=1)

            # l1 ball line search
            inner_t = 0
            while log_loss_k > log_loss_0 - c1 * alpha * normG**2:
                alpha = alpha * rho                             # change step sizee
                K = K_old - alpha*G 
                if np.linalg.norm(K[:], ord=1) > r:             # project onto L1 ball
                    K *= r / np.linalg.norm(K[:], ord=1)
                emp_loss_k, log_loss_k = lossK(K, X, S)
                inner_t += 1
                if inner_t > 10:
                    break

        else:
            K = projected(K, d)

            # PSD cone line search
            inner_t = 0
            while log_loss_k > log_loss_0 - c1 * alpha * normG**2:
                alpha = alpha * rho
                K = projected(K_old - alpha*G, d)
                emp_loss_k, log_loss_k = lossK(K, X, S)
                inner_t += 1
                if inner_t > 10:
                    break
        alpha = 1.1*alpha
        dif = np.linalg.norm(K - K_old, ord='fro')**2

        if verbose:
            print("Iterate=%d, emp_loss=%f, log_loss=%f, dif=%f, backsteps=%d, alpha=%f"
                  % (t, emp_loss_k, log_loss_k, dif,inner_t,alpha))
    return K


if __name__ == "__main__":
    n, p, d = 50, 10, 5
    lam = 0.005					# regularization parameter
    pulls = 3000
    X = features(n, p)
    Ktrue = kernel(p, d)

    r = np.linalg.norm(Ktrue[:], ord=1) * 1.05		# true radius of the L1 ball * multiplicative term for slack

    # Ktrue = np.eye(p)
    S = triplets(Ktrue, X, pulls, noise=False)
    M = M_set(S, X)
    print(len(M), M[1].shape)

    Khat, emp_losses, log_losses = computeKernel(
        X, S, d, lam, maxits=100, verbose=True)
    Khat_am = alternatingMin(X, S, r, d, verbose=True)
    print(np.linalg.norm(Khat, ord='fro'))
    print('recovery error', np.linalg.norm(Khat-Ktrue, 'fro')**2/np.linalg.norm(Ktrue, 'fro')**2)
    
    # plot comparison
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.imshow(Ktrue, interpolation='nearest')
    ax1.set_title("True Kernel")
    ax2.imshow(Khat, interpolation='nearest')
    ax2.set_title("Estimated Kernel - Composed Proximal gradients")
    ax3.imshow(Khat_am, interpolation='nearest')
    ax3.set_title("Estimated Kernel - Alternating Minimization")
    plt.show()
