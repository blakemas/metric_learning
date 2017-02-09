from __future__ import division, print_function

import numpy as np 
import matplotlib.pyplot as plt 

from utilsMetric import features, M_set, triplets

def varStat(Ms):
    mat = np.zeros(Ms[0].shape)
    for i in range(len(Ms)):
        e = np.sign(np.random.rand())
        mat += e * Ms[i]
    return np.linalg.norm(mat, ord=2)

def run(ps, n=100, pulls=1000, dim=2):
    var_high =[]        # variance stat of isotropic case
    var_low = []        # variance state of low case
    for m, p in enumerate(ps):
        print("Performing test %d of %d for p=%d" %(m+1, len(ps), p))
        K = np.eye(p)		# simple kernel for ease

        # generate isotropic X
        X_high = np.random.randn(n,p)
        row_norms = np.linalg.norm(X_high, axis=1)
        X_high[:] /= row_norms[:]      # scale all points to be of unit norm
        
        # generate X_low
        U, s, V = np.linalg.svd(X_high, full_matrices=False)
        s[dim:] = 0		# take top #dim singular values
        X_low = np.dot(U, np.dot(np.diag(s), V)) + 1e-3 * np.random.randn(n,p)
        print(np.linalg.cond(X_low))
        row_norms = np.linalg.norm(X_low, axis=1)
        X_low[:] /= row_norms[:]        # scale all points to be on sphere
        S = triplets(K, X_low, pulls, noise=False)
        var_low.append(varStat(M_set(S, X_low)))
        var_high.append(varStat(M_set(S, X_high)))
    return var_high, var_low



if __name__ == "__main__":	
    ps = np.arange(10,100,10)       # different number of features to try
    pulls = 1000
    n = 100         # number of points
    dim = 1         # what dimension the points approximately lie in
    var_high, var_low = run(ps, pulls=pulls, n=n)

    plt.plot(ps, var_high, label="Isotropic")
    plt.plot(ps, var_low, label="Low dimensional")
    plt.legend(loc='best')
    plt.xlabel('Number of features')
    plt.ylabel('Concetration')
    plt.title('Matrix concetration for high and low dimensional feature sets')
    plt.show()




