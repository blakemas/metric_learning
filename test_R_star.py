from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import json, uuid, sys, io
import cPickle as pickle

from scipy.linalg import orth
from utilsMetric import computeKernel, triplets, norm_nuc, norm_L12


def dense_case(n, d, p):
    U = orth(np.random.randn(p, d)) 
    Ktrue = np.dot(U, U.T)  * p/np.sqrt(d)
    X = np.random.randn(n, p) * 1./p**.5
    return Ktrue, X

def R_star(n, d, p):
    Ktrue, X = dense_case(n,d,p)
    total = 0
    R_star = 0
    pTrue = np.zeros((n, n, n))
    Gtrue = np.dot(X, np.dot(Ktrue, X.T))
    for i in range(n):
        for j in range(i):
            for k in range(n):
                if i != j and i != k and j != k:
                    score = Gtrue[k, k] - 2 * Gtrue[i,k] + 2 * Gtrue[i, j] - Gtrue[j, j]
                    pp = 1 / (1 + np.exp(-score))
                    pTrue[i, j, k] = pp
                    R_star += -pp * np.log(pp)
                    total += 1
    return R_star / total

def test():
    d = np.arange(2,11)
    d = [20]
    p = 100
    n = 110
    R_stars = []

    for i in range(len(d)):
        amt = 0
        dim = d[i]
        print(dim)
        for _ in range(10):
            amt += R_star(n, dim, p)
        R_stars.append(amt / 10)


    print(R_stars)
    log_coeffs = np.polyfit(np.log(d), np.log(R_stars), 1)
    slope = log[0]
    log_fit = [log_coeffs[0]*d + log_coeffs[1] for d in np.log(d)]
    plt.plot(log_fit)
    plt.show()

if __name__ == '__main__':
    test()



