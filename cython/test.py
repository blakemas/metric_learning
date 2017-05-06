from __future__ import division

import time
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from utilsMetric import *

n, p, d = 50, 20, 6

# A = np.array([[0,0,0],[4.,5.,6.],[0,0,0]])
# print(np.sum(np.linalg.norm(A, axis=1)))
# A = project_L12(A, 2)
# print(A)
# print(np.sum(np.linalg.norm(A, axis=1)))

Ktrue = np.eye(p)
for i in range(d, p):
    Ktrue[i, i] = 0
X = features(n, p, scale=d**.25)
lam = norm_L12(Ktrue)

num_samples = 5000
S = triplets(Ktrue, X, num_samples, noise=True)

ts = time.time()
Khat_nuc, emp_loss, log_loss = computeKernel(X, S, d, lam, 
                                         maxits=100, 
                                         epsilon=1e-3, 
                                         regularization='norm_L12', 
                                         verbose=True)
print 'time naive gd:', time.time() - ts

ts = time.time()
Khat_L12, emp_loss, log_loss = computeKernelEpochSGD(X, S, d, lam,
                                                 regularization='norm_L12', 
                                                 a=100,
                                                 maxits_sgd=1000000,
                                                 maxits_gd=10, 
                                                 epsilon=1e-3, 
                                                 verbose=True)
print 'time sgd:', time.time() - ts


# R_star = 0
# total = 0
# pTrue = np.zeros((n, n, n))
# Gtrue = np.dot(X, np.dot(Ktrue, X.T))
# for i in range(n):
#     for j in range(i):
#         for k in range(n):
#             if i != j and i != k and j!=k:
#                 pp = 1/(1+np.exp(-(Gtrue[k,k] -2*Gtrue[i,k] + 2*Gtrue[i,j] - Gtrue[j,j])))
#                 pTrue[i,j,k] = pp    
#                 R_star += -pp*np.log(pp)
#                 total += 1
# R_star = R_star/total
# print 'R_star', R_star


#Ms = M_set(S, X)
#bayes_emp_loss, bayes_log_loss = getLoss(Ktrue, Ms)

#print('Ktrue log loss: {} '.format(bayes_log_loss))
#print('Ktrue emp loss: {} '.format(bayes_emp_loss))

# Khat, emp_loss, log_loss = computeKernel(X, S, d, lam, 
#                                          maxits=100, 
#                                          epsilon=1e-5, 
#                                          regularization='norm_L12', 
#                                          verbose=True)
# R_hat = 0
# total = 0
# Ghat = np.dot(X, np.dot(Khat, X.T))
# for i in range(n):
#     for j in range(i):
#         for k in range(n):
#             if i != j and i != k and j!=k:
#                 pp = pTrue[i,j,k] 
#                 R_hat += pp*np.log(1+np.exp(-(Ghat[k,k] -2*Ghat[i,k] + 2*Ghat[i,j] - Ghat[j,j])))
#                 total += 1
# R_hat = R_hat/total
# print 'R_hat', R_hat
# print('Excess Risk: ', R_hat - R_star) 


# bayes_emp_loss, bayes_log_loss = getLoss(Ktrue, Ms)
# Khat_emp_loss, Khat_log_loss = getLoss(Khat, Ms)
# print('Ktrue log loss: {} Khat Log_loss: {}'.format(bayes_log_loss, Khat_log_loss))
# print('Ktrue emp loss: {} Khat emp_loss: {}'.format(bayes_emp_loss, Khat_emp_loss))
# #print('Ktrue eig',Ktrue) 
# #print('Khat eig',Khat) 
# rel_err = np.linalg.norm(Ktrue - Khat, ord='fro')**2/np.linalg.norm(Ktrue, ord='fro')**2
#print('Recovery error: ', rel_err)

plt.figure(1)
plt.subplot(121)
plt.imshow(Khat_nuc, interpolation='none')
plt.subplot(122)
plt.imshow(Khat_L12, interpolation='none')
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
