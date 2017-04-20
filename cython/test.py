from __future__ import division, print_function

import numpy as np 
import matplotlib.pyplot as plt
import blackbox
from utilsMetric import *

n, p, d = 200, 8, 2
# lam  = 0.25             # regularization

# A = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
# print(np.sum(np.linalg.norm(A, axis=1)))
# A = L12_projection(A, 20)
# print(A)
# print(np.sum(np.linalg.norm(A, axis=1)))

X = features(n, p)
Ktrue = kernel(p, d)
lam = 2*norm_L12(Ktrue)
num_samples = 1000
S = triplets(Ktrue, X, num_samples, noise=True)

blackbox.set_experiment("test_data")
blackbox.takeoff(("Test_experiment"), force=True)
Khat, emp_loss, log_loss = computeKernel(X, S, d, lam, 
                                        maxits=100, 
                                        epsilon=1e-3, 
                                        regularization='alternating', 
                                        verbose=True)
blackbox.land()
Ms = M_set(S, X)
bayes_emp_loss, bayes_log_loss = getLoss(Ktrue, Ms)
Khat_emp_loss, Khat_log_loss = getLoss(Khat, Ms)
print('Ktrue log loss: {} Khat Log_loss: {}'.format(bayes_log_loss, Khat_log_loss))
print('Ktrue emp loss: {} Khat emp_loss: {}'.format(bayes_emp_loss, Khat_emp_loss))

rel_err = np.linalg.norm(Ktrue - Khat, ord='fro')**2/np.linalg.norm(Ktrue, ord='fro')**2
print('Recovery error: ', rel_err)



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
