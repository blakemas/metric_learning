from __future__ import division, print_function
from time import time
import numpy as np

import tensorflow as tf 
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
lam1 = tf.constant(0.05)
# regularization parameter for L_{1,2} cost
lam12 = tf.constant(0.05)
# n p-dimensional feature vectors
X = features(n, p)
# The true Kernel, a p by p symmetric PSD kernel with d^2 non-zero entries
Ktrue = kernel(p, d)
print('Ktrue shape', np.shape(Ktrue))
pulls = 5000                                    # number of triplets gathered
S = triplets(Ktrue, X, pulls, noise=True)         # get some triplets
pulls = tf.constant(pulls, dtype=tf.float32)
Ms = [tf.constant(M.T.astype('float32')) for M in M_set(S, X)]        # all different Mts stored at tf constants


# Instantiate a manifold
manifold = PositiveDefinite(p)

##### L1 loss
K = tf.Variable(tf.placeholder(tf.float32, [p, p]))
Ker = tf.matmul(K, tf.transpose(K))
costL1 = tf.add_n([tf.log1p(tf.exp(-1.*tf.trace(tf.matmul(Ker, M)))) \
                                                        for M in Ms])/pulls \
                                                        + lam1 * tf.reduce_sum(tf.abs(Ker))

##### L12 loss
costL12 = tf.add_n([tf.log1p(tf.exp(-1.*tf.trace(tf.matmul(Ker, M)))) \
                                                        for M in Ms])/pulls \
                                                        + lam12 * tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(Ker), 0)))

# create the problems, defined over the manifold
problem_L1 = Problem(manifold=manifold, cost=costL1, arg=K)
problem_L12 = Problem(manifold=manifold, cost=costL12, arg=K)
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
