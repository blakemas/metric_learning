from __future__ import division, print_function

import numpy as np 
import matplotlib.pyplot as plt 

from utilsMetric import *
import blackbox


n, p, d = 50, 10, 5
lam_L12 = 0.01					# regularization parameter
lam_nucNorm = 0.05
pulls = 3000
X = features(n, p)
Ktrue = kernel(p, d)
np.random.seed(42)
S = triplets(Ktrue, X, pulls, noise=False)

blackbox.set_experiment("Metric Learning")

blackbox.takeoff(("L12: n=50, p=10, d=5, lam_L12=0.01"), force=True)
print("Computing with L12 regularization")
Khat_L12, emp_losses_L12, log_losses_L12 = computeKernel(
    X, S, d, lam_L12, maxits=1000, epsilon=1e-6, regularization='L12', verbose=True)
blackbox.land()

blackbox.takeoff(("Nuclear Norm: n=50, p=10, d=5, lam_nucNorm=0.06"), force=True)
print("Computing with Nuclear Norm regularization")
Khat_nucNorm, emp_losses_nucNorm, log_losses_nucNorm = computeKernel(
    X, S, d, lam_nucNorm, maxits=1000, epsilon=1e-6, regularization='nucNorm', verbose=True)
blackbox.land()


# plot comparison
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.imshow(Ktrue, interpolation='nearest')
ax1.set_title("True Kernel")
ax2.imshow(Khat_L12, interpolation='nearest')
ax2.set_title("Estimated Kernel - L_12 regularization")
ax3.imshow(Khat_nucNorm, interpolation='nearest')
ax3.set_title("Estimated Kernel - Nuclear Norm regularization")
plt.show()

