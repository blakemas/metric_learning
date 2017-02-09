from __future__ import division, print_function

import numpy as np 
import matplotlib.pyplot as plt 

from utilsMetric import *
import blackbox


n, p, d = 50, 10, 5
lam_L12 = 0.01					# regularization parameter
lam_nucNorm = 0.05
lam_L1 = 0.01
X = features(n, p)
Ktrue = kernel(p, d)
np.random.seed(42)
complexity = int(d**2 * np.log(2*p))

blackbox.set_experiment("Silo images")


blackbox.takeoff(("L12: n=100, p=50, d=20, lam_L12=0.01"), force=True)
pulls = int(5* complexity)
S = triplets(Ktrue, X, pulls, noise=False)
print("Computing with L12 regularization - 2*complexity")
Khat_2, emp_losses_2, log_losses_2 = computeKernel(
    X, S, d, lam_nucNorm, maxits=200, epsilon=1e-6, regularization='nucNorm', verbose=True)
blackbox.land()

blackbox.takeoff(("L12: n=100, p=50, d=20, lam_L12=0.01"), force=True)
pulls = int(10* complexity)
S = triplets(Ktrue, X, pulls, noise=False)
print("Computing with L12 regularization - 5*complexity")
Khat_5, emp_losses_5, log_losses_5 = computeKernel(
    X, S, d, lam_nucNorm, maxits=200, epsilon=1e-6, regularization='nucNorm', verbose=True)
blackbox.land()

blackbox.takeoff(("L12: n=100, p=50, d=20, lam_L12=0.01"), force=True)
pulls = int(20* complexity)
S = triplets(Ktrue, X, pulls, noise=False)
print("Computing with L12 regularization - 5*complexity")
Khat_10, emp_losses_10, log_losses_10 = computeKernel(
    X, S, d, lam_nucNorm, maxits=200, epsilon=1e-6, regularization='nucNorm', verbose=True)
blackbox.land()

# plot comparison
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)  #, sharex='col', sharey='row')
ax1.imshow(Ktrue, interpolation='nearest')
ax1.set_title("True Kernel")
ax2.imshow(Khat_2, interpolation='nearest')
ax2.set_title("5*d^2*log(p)")
ax3.imshow(Khat_5, interpolation='nearest')
ax3.set_title("10*d^2*log(p)")
ax4.imshow(Khat_10, interpolation='nearest')
ax4.set_title("20*d^2*log(p)")
plt.show()