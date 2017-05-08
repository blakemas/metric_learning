from __future__ import division, print_function

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import savemat

from utilsMetric import project_L12, alternating_projection

def L12_norm(M):
    return np.sum(np.sqrt(np.abs(np.sum(M*M, 1))));

n, d, lam = 10, 5, 10.
M = np.random.randn(n,d)
M = M.dot(M.T)

# M = np.diag(np.random.rand(n))


# print(L12_norm(M))
M_old = M.copy()
M_proj = project_L12(M.copy(), lam)

savemat('../matlab_L12/data.mat', {'n':n, 'd':d, 'lam':lam, 'Mtrue':M_old, 'Mproj':M_proj})

### checking the bouncing projection
M = np.random.randn(n,n)
M = M + M.T
# w, _ = np.linalg.eig(M)
# print(w)
lam = 15
bounce = 2
Mhat = alternating_projection(M, lam, bounce)
w, _ = np.linalg.eig(Mhat)
print(L12_norm(Mhat), np.real(w))
