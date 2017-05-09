from __future__ import division
import numpy as np
import json, uuid, sys, io
from collections import defaultdict
import multiprocessing as mp
from Queue import Queue
import cPickle as pickle
from dask.distributed import Client

import msgpack
import msgpack_numpy as mn
mn.patch()

client = Client()#'34.201.36.27:8786')
#client.upload_file('dist/utilsMetric-0.0.0-py2.7-linux-x86_64.egg')
from scipy.linalg import orth
from utilsMetric import computeKernel, triplets, norm_nuc


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

# def dense_case(n, d, p):
#     U = orth(np.random.randn(p, d))
#     Ktrue = np.dot(U, U.T)
#     X = features(n, p, scale=d**.25)
#     return Ktrue, X

def learn_metric_helper(args):
    return learn_metric(*args)

def learn_metric(n, d, p, seed, step=5000, start=5000, acc=.01):
    id = np.random.randint(1000)
    np.random.seed(seed)
    Ktrue, X = sparse_case(n, d, p)

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
    print('id:{}, true Bayes error: {}, p:{}, d:{}'.format(id, R_star, p, d))

    # List of relative errors per iteration
    rel_err_list = []
    loss_list = []
    S = triplets(Ktrue, X, start, noise=True)
    Ks = [(Ktrue, Ktrue)]
    it = 0
    rel_err_nuc = float('inf')
    rel_err_L12 = float('inf')

    while max(rel_err_L12, rel_err_nuc) > acc:
        it += 1
        if rel_err_nuc > acc:
            Khat_nuc, emp_loss, log_loss = computeKernel(X, S, d,
                                                         norm_nuc(Ktrue),
                                                         maxits=500,
                                                         epsilon=1e-5,
                                                         regularization='norm_nuc',
                                                         verbose=False)
            rel_err_nuc, loss_nuc = comparative_risk(R_star, Khat_nuc, X, pTrue)
        if rel_err_L12 > acc:
            Khat_L12, emp_loss, log_loss = computeKernel(X, S, d,
                                                         norm_nuc(Ktrue),
                                                         maxits=500,
                                                         epsilon=1e-5,
                                                         regularization='norm_L12',
                                                         verbose=False)
            rel_err_L12, loss_L12 = comparative_risk(R_star, Khat_L12, X, pTrue)

        rel_err_list.append((rel_err_nuc, rel_err_L12))
        loss_list.append((loss_nuc, loss_L12))
        print(("id:{}. Current relative error: ({:.6f}, {:.6f}), log_losses: ({:.6f}, {:.6f}), "
               "New test of {} samples. Dimension:{}. Sparsity:{}. Iteration: {}").format(id,
                                                                                               rel_err_nuc, rel_err_L12,
                                                                                               loss_nuc, loss_L12,
                                                                                               len(S), p, d, it))
        S.extend(triplets(Ktrue, X, step, noise=True))
        Ks.append((Khat_nuc, Khat_L12))
    result = {'pulls': len(S), 'Ks': Ks, 'n': n, 'd': d, 'p': p, 'start': start, 'step': step, 'X': X,
              'rel_err_list': rel_err_list, 'loss_list': loss_list}
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
    result_q = client.map(learn_metric_helper, remote_q)
    gather_q = client.gather(result_q)
    map(input_q.put, inputs)
    stream = io.open(stream_name,'wb', buffering=0)
    while True:
        data  = gather_q.get()
        stream.write(msgpack.packb(data))
        print('finished task: n-{}, d-{}, p-{}'.format(data['n'], data['d'], data['p']))
    return results



if __name__ == '__main__':
    if sys.argv[1] == 'test':
        d = [2, 4, 6, 8]  # , 8, 10, 12, 14, 16, 18, 20]
        step = [250] * len(d)
        start = [250] * len(d)
        p = [10] * len(d)
        n = [15] * len(d)
        acc = .9
        avg = 1        # number of runs to average over
        results = driver(n, d, p, step,
                         start, avg=avg, acc=acc, stream_name='test-dump.dat')    
    else:
        d = [5, 10, 15, 20, 25, 30, 35, 40, 45]  # , 8, 10, 12, 14, 16, 18, 20]
        step = [500, 500, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        start = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
        p = [50] * len(d)
        n = [60] * len(d)
        acc = .1
        avg = 20        # number of runs to average over
        results = driver(n, d, p, step,
                         start, avg=avg, acc=acc,
                         stream_name='sparse-results-n{}-d{}-p{}-acc{}-avg{}.dat'.format(n, d, p, acc, avg))
        pickle.dump(results,
                    open('sparse-results-n{}-d{}-p{}-acc{}-avg{}.pkl'.format(n,d,p,acc,avg), 'wb'))
        
