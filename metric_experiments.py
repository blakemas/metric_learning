from __future__ import division
import numpy as np
import json, uuid, sys, io
from collections import defaultdict
import multiprocessing as mp
from Queue import Queue
import cPickle as pickle
from dask.distributed import Client

import matplotlib.pyplot as plt

import msgpack
import msgpack_numpy as mn
mn.patch()

client = Client()
#client.upload_file('cython/dist/utilsMetric-0.0.0-py2.7-linux-x86_64.egg')
from scipy.linalg import orth
from utilsMetric import computeKernel, triplets, norm_nuc, norm_L12


def comparative_risk(R_star, K, X, pTrue):
    R_hat = 0
    total = 0
    n = X.shape[0]
    G = np.dot(X, np.dot(K, X.T))
    for i in range(n):
        for j in range(i):
            for k in range(n):
                if i != j and i != k and j != k:
                    score = -(G[k, k] - 2 * G[i, k] + 2 * G[i, j] - G[j, j])
                    R_hat += pTrue[i, j, k] * np.log(1 + np.exp(score))
                    total += 1
    R_hat = R_hat / total
    return (R_hat - R_star)/R_star, R_hat


def sparse_case(n, d, p):
    Ktrue = np.eye(p)
    for i in range(d, p):
        Ktrue[i,i] = 0
    X = np.random.randn(n, p) * 1/d**.25
    return Ktrue, X

def dense_case(n, d, p):
    U = orth(np.random.randn(p, d))
    Ktrue = np.dot(U, U.T)
    X = np.random.randn(n, p) * 1/d**.25
    return Ktrue, X

def row_sparse_case(n, d, p):
    subK = np.random.randn(d, d)*p**.5/d**.75
    subK = np.dot(subK.T, subK)
    inds = np.arange(p)
    np.random.shuffle(inds)
    inds = inds[:d]             # get d random indicies
    K = np.zeros((p, p))
    K[[[i] for i in inds], inds] = subK

    X = np.random.randn(n, p) * 1/p**.5
    return K, X

def learn_metric_helper(args):
    return learn_metric(*args)

def learn_metric(args):
    n, d, p, seed, step, start, acc = args
    id = np.random.randint(1000)
    np.random.seed(seed)
    Ktrue, X = dense_case(n, d, p)
    # Compute the true risk
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
    R_star = R_star / total

    # List of relative errors per iteration
    pred_err_list = []
    loss_list = []
    rec_err_list = []
    S = triplets(Ktrue, X, start, noise=True)
    Ks = [(Ktrue, Ktrue)]
    it = 0
    pred_err_nuc = float('inf')
    pred_err_L12 = float('inf')

    rec_err_nuc = float('inf')
    rec_err_L12 = float('inf')

    print('id:{}, true Bayes error: {}, p:{}, d:{}, |S|:{}'.format(id, R_star, p, d, len(S)))
    while max(pred_err_L12, pred_err_nuc) > acc:
        it += 1
        if pred_err_nuc > acc:
            Khat_nuc, emp_loss, log_loss = computeKernel(X, S, d,
                                                         norm_nuc(Ktrue),
                                                         maxits=500,
                                                         epsilon=1e-6,
                                                         c1 = 1e-4,
                                                         regularization='norm_nuc',
                                                         verbose=True)
            pred_err_nuc, loss_nuc = comparative_risk(R_star, Khat_nuc, X, pTrue)
            rec_err_nuc = np.linalg.norm(Khat_nuc-Ktrue,'fro')**2/np.linalg.norm(Ktrue,'fro')**2
        if pred_err_L12 > acc:
            Khat_L12, emp_loss, log_loss = computeKernel(X, S, d,
                                                         norm_L12(Ktrue),
                                                         maxits=500,
                                                         epsilon=1e-6,
                                                         c1 = 1e-4,
                                                         regularization='norm_L12',
                                                         verbose=True)
            pred_err_L12, loss_L12 = comparative_risk(R_star, Khat_L12, X, pTrue)
            rec_err_L12 = np.linalg.norm(Khat_L12-Ktrue,'fro')**2/np.linalg.norm(Ktrue,'fro')**2
        pred_err_list.append((pred_err_nuc, pred_err_L12))
        rec_err_list.append((rec_err_nuc, rec_err_L12))
        loss_list.append((loss_nuc, loss_L12))
        print(("id:{}. Current prediction error: ({:.6f}, {:.6f}), Current recovery error: ({:.6f}, {:.6f}), log_losses: ({:.6f}, {:.6f}), "
               "New test of {} samples. Dimension:{}. Sparsity:{}. Iteration: {}").format(id,
                                                                                          pred_err_nuc, pred_err_L12,
                                                                                          rec_err_nuc, rec_err_L12,
                                                                                          loss_nuc, loss_L12,
                                                                                          len(S)+step, p, d, it))
        print('After learning: norm_nuc', norm_nuc(Khat_nuc), 'norm_L12', norm_L12(Khat_L12))
        S.extend(triplets(Ktrue, X, step, noise=True))
        Ks.append((Khat_nuc, Khat_L12))
    result = {'pulls': len(S), 'Ks': Ks, 'n': n, 'd': d, 'p': p, 'start': start, 'step': step, 'X': X, 'pred_err_list':pred_err_list,
              'rec_err_list': rec_err_list, 'loss_list': loss_list}
    return result


def driver(n, d, p, step, start, avg=3, acc=0.01, stream_name='stream'):
    seed = np.random.randint(1000)
    print(seed)
    input_q = Queue()
    inputs = [(n[int(i / avg)],
               d[int(i / avg)],
               p[int(i / avg)],
               seed + i,
               step[int(i / avg)],
               start[int(i / avg)],
               acc)
              for i in range(avg * len(d))]
    remote_q = client.scatter(input_q)
    result_q = client.map(learn_metric, remote_q)
    gather_q = client.gather(result_q)
    map(input_q.put, inputs)
    stream = io.open(stream_name,'wb', buffering=0)
    total = 0
    results  = []
    while total < avg*len(d):
        data  = gather_q.get()
        total += 1
        stream.write(msgpack.packb(data))
        print('finished task: n-{}, d-{}, p-{}'.format(data['n'], data['d'], data['p']))
        results.append(data)
    return results



if __name__ == '__main__':
    if sys.argv[1] == 'test':
        d = [25]  # , 8, 10, 12, 14, 16, 18, 20]
        step = [50000] * len(d)
        start = [100000] * len(d)
        p = [50] * len(d)
        n = [50] * len(d)
        acc = .01
        avg = 1        # number of runs to average over
        Ktrue, X = dense_case(n[0],d[0],p[0])
        plt.imshow(Ktrue, interpolation='none')
        plt.show()
        print('norm_nuc', norm_nuc(Ktrue), 'norm_L12', norm_L12(Ktrue))

        results = driver(n, d, p, step,
                         start, avg=avg, acc=acc, stream_name='test-dump.dat')    
        pickle.dump(results, open('test-dump.pkl'.format(n,d,p,acc,avg), 'wb'))
        
    else:
        d = [3, 3, 3, 3, 3, 3, 3, 3, 3]  
        step = [100, 100, 250, 250, 500, 500, 500, 500, 500]
        start = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 2500]
        p = [10, 15, 20, 25, 30, 35, 40, 45, 50]
        n = [60] * len(d)
        acc = .1
        avg = 20        # number of runs to average over
        results = driver(n, d, p, step,
                         start, avg=avg, acc=acc,
                         stream_name='dense-results-n{}-d{}-p{}-acc{}-avg{}.dat'.format(n, d, p, acc, avg))
        pickle.dump(results,
                    open('dense-results-n{}-d{}-p{}-acc{}-avg{}.pkl'.format(n,d,p,acc,avg), 'wb'))
        
