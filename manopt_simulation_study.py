from __future__ import division, print_function
from time import time

import autograd.numpy as np
from autograd.core import primitive
from pymanopt.manifolds import PSDFixedRank, PositiveDefinite
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, ConjugateGradient
from pymanopt.solvers.linesearch import LineSearchBackTracking

from utilsMetric import *

# experiment parameters
# make variables for cost computations global becuase only can pass one
# argument to cost functions
global X, S, lam1, lam12
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
print('Ktrue shape', np.shape(Ktrue))
pulls = 5000                                    # number of triplets gathered
S = triplets(Ktrue, X, pulls, noise=True)       # get some triplets

# Instantiate a manifold
manifold = PositiveDefinite(p)
# Define the cost function for the L_1 and L_{1,2} problems (here using
# autograd.numpy)

@primitive
def norm1(x):
    return np.sum(np.abs(x))


def make_grad_norm1(ans, x):
    def gradient_product(g):
        s = np.sign(x)
        return np.full(x.shape, g)*s
    return gradient_product
norm1.defgrad(make_grad_norm1)


def costL1(K):
    loss = 0.
    Ker = np.dot(K, K.T)
    for q in S:
        i, j, k = q
        Mt = (2. * np.outer(X[i], X[j]) - 2. * np.outer(X[i], X[k])
              - np.outer(X[j], X[j]) + np.outer(X[k], X[k]))
        loss = loss + np.log(1. + np.exp(-np.trace(np.dot(Mt, Ker))))
    # add L1 penalty
    loss = loss / len(S) + lam1 * norm1(Ker.flatten())
    return loss


def costL12(K):
    # compute log loss
    loss = 0.
    for q in S:
        i, j, k = q
        Mt = (2. * np.outer(X[i], X[j]) - 2. * np.outer(X[i], X[k])
              - np.outer(X[j], X[j]) + np.outer(X[k], X[k]))
        loss = loss + np.log(1 + np.exp(-1. * np.trace(np.dot(Mt, K))))
    loss = loss

    # compute 1,2 norm of K
    norm12 = 0.
    for i in range(K.shape[0]):
        norm12 += np.linalg.norm(K[i])

    loss = loss / len(S) + lam12 * norm12
    return loss

# create the problems, defined over the manifold
problem_L1 = Problem(manifold=manifold, cost=costL1)
problem_L12 = Problem(manifold=manifold, cost=costL12)
# Instantiate a pymanopt solver
solver = ConjugateGradient(maxiter=100)

# solve each problem:

# Lasso method:
print("Beginning test with L_1")
ts = time()
Khat_L1 = solver.solve(problem_L1)
print(np.shape(Khat_L1))
tot_time = time() - ts
emp_loss, log_loss = lossK(Khat_L1, X, S)
print("L_1 regularization: Time=%f, emp_loss=%f, log_loss=%f" %
      (tot_time, emp_loss, log_loss))

# L_{1,2} regularized method
print("Beginning test with L_{1,2}")
ts = time()
Khat_L12 = solver.solve(problem_L12)
tot_time = time() - ts
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
