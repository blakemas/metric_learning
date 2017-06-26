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

# client = Client('54.164.43.247:8786')
# client.upload_file('cython/dist/utilsMetric-0.0.0-py2.7-linux-x86_64.egg')
client = Client()
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
    Ktrue = p/np.sqrt(d)*np.dot(U, U.T)
    X = np.random.randn(n, p) * 1/p**.5
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
    n, d, p, seed, triples = args
    np.random.seed(seed)
    id = np.random.randint(1000)
    Ktrue, X = row_sparse_case(n, d, p)

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
    S = triplets(Ktrue, X, triples, noise=True)
    Ks = [Ktrue]
    it = 0

    print('id:{}, true Bayes error: {}, p:{}, d:{}, |S|:{}'.format(id, R_star, p, d, len(S)))
    print('norm_nuc', norm_nuc(Ktrue), 'norm_L12', norm_L12(Ktrue), 'expected norm', np.sqrt(d)*p)

    Khat_psd, emp_loss, log_loss = computeKernel(X, S, d,
                                                 norm_nuc(Ktrue),
                                                 maxits=800,
                                                 epsilon=1e-8,
                                                 c1 = 1e-8,
                                                 alpha = 0.1,
                                                 regularization='psd',
                                                 verbose=True)

    pred_err_psd, loss_psd = comparative_risk(R_star, Khat_psd, X, pTrue)
    rec_err_psd = np.linalg.norm(Khat_psd-Ktrue,'fro')**2/np.linalg.norm(Ktrue,'fro')**2


    pred_err_list.append(pred_err_psd)
    rec_err_list.append(rec_err_psd)
    loss_list.append(loss_psd)
    print(("id:{}. Current prediction error: {:.6f}, Current recovery error: {:.6f}, log_losses: {:.6f}, "
           "Test of {} samples.Dimension:{}. Sparsity:{}. Iteration: {}").format(id,
                                                                                 pred_err_psd,
                                                                                 rec_err_psd,
                                                                                 loss_psd,
                                                                                 len(S), p, d, it))
    print('learned norm_nuc', norm_nuc(Ktrue), 'learned norm_L12', norm_L12(Khat_psd))
    Ks.append(Khat_psd)
    result = {'triples': len(S), 'Ks': Ks, 'n': n, 'd': d, 'p': p, 'X': X, 'pred_err_list':pred_err_list, 'seed':seed,
              'rec_err_list': rec_err_list, 'loss_list': loss_list, 'id':id}
    return result


def driver(n, d, p, seeds, triples, stream_name='stream'):
    input_q = Queue()
    inputs = [(n[i],
               d[i],
               p[i],
               seeds[i],
               triples[i])
              for i in range(len(triples))]
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
        print('finished task: n-{}, d-{}, p-{}, |S|-{}'.format(data['n'], data['d'], data['p'], data['triples']))
        results.append(data)
    return results



if __name__ == '__main__':
    if sys.argv[1] == 'test':
        avg = 1
        seeds = [42]
        d = [10]
        p = [100]
        n = [200]
        triples = [80000]
        results = driver(n, d, p, seeds, triples, stream_name='test_stream')
        Khat = results[0]['Ks'][-1]
        Ktrue = results[0]['Ks'][0]

        evals_hat, _ = np.linalg.eig(Khat)
        evals_true, _ = np.linalg.eig(Ktrue)

        plt.figure(1)
        plt.plot(np.sort(evals_true)[::-1], 'b', label='Ktrue')
        plt.plot(np.sort(evals_hat)[::-1], 'r', label='Khat')
        plt.legend()
        plt.title('Eigenvalues of Khat vs Ktrue - 80k samples')
        plt.show()
        # pickle.dump(results, open('test-dump.pkl', 'wb'))

    else:
        avg = 2        # number of runs to average over
        triples = []
        seeds = []
        
        start_seeds = []
        for i in range(avg):
            start_seeds.append(np.random.randint(10000))

        for i in [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]:
            for j in range(avg):
                triples.append(i)
                seeds.append(start_seeds[j])
            
        d = [10]*len(triples)
        p = [100]*len(triples)
        n = [200]*len(triples)

        rand_id = str(uuid.uuid4())[:10]
        results = driver(n, d, p, seeds, triples,
                         stream_name='run1_exp5-results-avg{}-id{}.dat'.format(avg, rand_id))
        pickle.dump(results,
                    open('run1_exp5-results-avg{}-id{}.pkl'.format(avg, rand_id), 'wb'))
        
