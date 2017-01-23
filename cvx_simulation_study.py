from __future__ import division, print_function
from time import time

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

from utilsMetric import *

# experiment parameters
# global X, S, lam1, lam12                        # make variables for
# cost computations global becuase only can pass one argument to cost
# functions
# number of points, number of total features, number of relevant features
n, p, d = 50, 10, 5
# regularization parameter for L_1 cost
lam1 = 0.05
# regularization parameter for L_{1,2} cost
lam12 = 0.05
# n p-dimensional feature vectors
X = features(n, p)
# The true Kernel, a p by p symmetric PSD kernel with d^2 non-zero entries
Ktrue = kernel(p, d)
pulls = 5000                                    # number of triplets gathered
S = triplets(Ktrue, X, pulls, noise=True)       # get some triplets

# Define the cost function for the L_1 and L_{1,2} problems (here using
# autograd.numpy)


def costL1(K, X, S, lam):
    # print(type(K))
    loss = 0.
    for q in S:
        i, j, k = q
        # should this be Mt^T * K??
        Mt = (2. * np.outer(X[i], X[j]) - 2. * np.outer(X[i], X[k]) - np.outer(
            X[j], X[j]) + np.outer(X[k], X[k]))
        score = cvx.trace(Mt.T * K)
        loss = loss + cvx.logistic(-score)
    loss = loss / len(S) + lam * cvx.norm(K, 1)      # add L1 penalty
    return loss


def costL12(K, X, S, lam):
    # compute log loss
    loss = 0.
    for q in S:
        i, j, k = q
        # should this be Mt^T * K??
        Mt = (2. * np.outer(X[i], X[j]) - 2. * np.outer(X[i], X[k]) - np.outer(
            X[j], X[j]) + np.outer(X[k], X[k]))
        score = cvx.trace(Mt.T * K)
        loss = loss + cvx.logistic(-score)
    # regularize with the 1,2 norm
    loss = loss / len(S) + lam * cvx.mixed_norm(K, 2, 1)
    return loss

# create L1 problem
K = cvx.Semidef(p)
objectiveL1 = cvx.Minimize(costL1(K, X, S, lam1))
probL1 = cvx.Problem(objectiveL1)

# solve the L1 problemLasso method:
print("Beginning test with L_1")
ts = time()
probL1.solve(verbose=True)
tot_time = time() - ts
Khat_L1 = np.array(K.value)
emp_loss, log_loss = lossK(Khat_L1, X, S)
print("L_1 regularization: Time=%f, emp_loss=%f, log_loss=%f" %
      (tot_time, emp_loss, log_loss))

# create L_{1,2} problem
K = cvx.Semidef(p)
objectiveL12 = cvx.Minimize(costL12(K, X, S, lam12))
probL12 = cvx.Problem(objectiveL12)

# solve the L_{1,2} regularized method
print("Beginning test with L_{1,2}")
ts = time()
probL12.solve(verbose=True)
tot_time = time() - ts
Khat_L12 = np.array(K.value)
emp_loss, log_loss = lossK(Khat_L12, X, S)
print("L_{1,2} regularization: Time=%f, emp_loss=%f, log_loss=%f" %
      (tot_time, emp_loss, log_loss))

# Plot comparison of recovered kernels:
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.imshow(Ktrue, interpolation='nearest')
ax1.set_title("True Kernel")
ax2.imshow(Khat_L1, interpolation='nearest')
ax2.set_title("Estimated Kernel - L1 regularization")
ax3.imshow(Khat_L12, interpolation='nearest')
ax3.set_title("Estimated Kernel - L12 Regularization")
plt.show()
