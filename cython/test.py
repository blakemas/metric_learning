from __future__ import division, print_function

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from utilsMetric import *

n, p, d = 200, 8, 2

# A = np.array([[0,0,0],[4.,5.,6.],[0,0,0]])
# print(np.sum(np.linalg.norm(A, axis=1)))
# A = project_L12(A, 2)
# print(A)
# print(np.sum(np.linalg.norm(A, axis=1)))


X = features(n, p, scale=d**.25)
Ktrue = kernel(p, d, scale=d, sparse=True)
lam = norm_L12(Ktrue)

num_samples = 10000
S = triplets(Ktrue, X, num_samples, noise=True)
Khat, emp_loss, log_loss = computeKernel(X, S, d, lam, 
                                         maxits=100, 
                                         epsilon=1e-5, 
                                         regularization='norm_L12', 
                                         verbose=True)
Ms = M_set(S, X)
bayes_emp_loss, bayes_log_loss = getLoss(Ktrue, Ms)
Khat_emp_loss, Khat_log_loss = getLoss(Khat, Ms)
print('Ktrue log loss: {} Khat Log_loss: {}'.format(bayes_log_loss, Khat_log_loss))
print('Ktrue emp loss: {} Khat emp_loss: {}'.format(bayes_emp_loss, Khat_emp_loss))
print('Ktrue eig',Ktrue) 
print('Khat eig',Khat) 
rel_err = np.linalg.norm(Ktrue - Khat, ord='fro')**2/np.linalg.norm(Ktrue, ord='fro')**2
print('Recovery error: ', rel_err)

plt.figure(1)
plt.subplot(121)
plt.imshow(Ktrue, interpolation='none')
plt.subplot(122)
plt.imshow(Khat, interpolation='none')
plt.show()




# opvalues = []
# for p in range(2,200,10):
#     avg = 0
#     for i in range(10):
#         a = 0
#         X = features(n, p)
#     #X = #np.eye(p)
#         Ktrue = kernel(p, d)
#     # Ktrue = np.eye(p)
#         num_samples = 2000
#         S = triplets(Ktrue, X, num_samples, noise=True)
#         Ms = M_set(S, X)
#         for M in Ms:
#             a += np.abs(getScore(Ktrue, M))
#         avg = avg + a/num_samples
#     print(avg/10)
#     opvalues.append(avg/10)
#     #emp_loss, log_loss = getLoss(Ktrue, Ms)
#     #print('Prediction accuracy: {}, Log loss: {}'.format(emp_loss, log_loss))
# print(len(opvalues))
# plt.scatter(range(2,200,10), opvalues)
# plt.show()
