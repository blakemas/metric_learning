from __future__ import division
import json, uuid, sys
from collections import defaultdict
import multiprocessing as mp, io 

import msgpack
import msgpack_numpy as mn
mn.patch()

import numpy as np 
from scipy.optimize import curve_fit
from scipy.linalg import orth
import matplotlib.pyplot as plt 
from joblib import Parallel, delayed
from utilsMetric import *

def comparative_risk(R_star, K, X, pTrue):
    R_hat = 0
    total = 0
    n = X.shape[0]
    G = np.dot(X, np.dot(K, X.T))
    for i in range(n):
        for j in range(i):
            for k in range(n):
                if i!=j and i!=k and j!=k:
                    R_hat += pTrue[i,j,k]*np.log(1+np.exp(-(G[k,k] -2*G[i,k] + 2*G[i,j] - G[j,j])))
                    total += 1
    R_hat = R_hat/total
    return R_hat - R_star, R_hat

def estimate_pulls(n, d, p, seed, q, step=5000, acc=0.1, max_norm=1.):
    np.random.seed(seed)
    id = np.random.randint(1000)
    U = orth(np.random.randn(p,d))
    Ktrue = np.dot(U,U.T)
    X = features(n, p, scale = d**.25)
    rel_err_L12 = float('inf')
    rel_err_nuc = float('inf')
    total = 0
    R_star = 0
    pTrue = np.zeros((n, n, n))
    Gtrue = np.dot(X, np.dot(Ktrue, X.T))
    for i in range(n):
        for j in range(i):
            for k in range(n):
                if i != j and i != k and j!=k:
                    pp = 1/(1+np.exp(-(Gtrue[k,k] -2*Gtrue[i,k] + 2*Gtrue[i,j] - Gtrue[j,j])))
                    pTrue[i,j,k] = pp    
                    R_star += -pp*np.log(pp)
                    total += 1
    R_star = R_star/total
    rel_err_list = []
    loss_list = []
    print('id:{}, true Bayes error: {}, p:{}, d:{}'.format(id, R_star, p, d))
    S = triplets(Ktrue, X, 5000+step, noise=True)
    Ks = [(Ktrue,Ktrue)]
    it = 0                                      
    while max(rel_err_L12, rel_err_nuc) > acc:
        it += 1 
        if rel_err_nuc > acc:        
            Khat_nuc, emp_loss, log_loss = computeKernel(X, S, d, norm_nuc(Ktrue), 
                                                         maxits=100, 
                                                         epsilon=1e-5, 
                                                         regularization='norm_nuc',
                                                         verbose=False)
            rel_err_nuc, loss_nuc = comparative_risk(R_star, Khat_nuc, X, pTrue)
        if rel_err_L12 > acc:
            Khat_L12, emp_loss, log_loss = computeKernel(X, S, d, norm_nuc(Ktrue), #  
                                                         maxits=100, 
                                                         epsilon=1e-5, 
                                                         regularization='norm_L12',
                                                         verbose=False)
            rel_err_L12, loss_L12 = comparative_risk(R_star, Khat_L12, X, pTrue)
        rel_err_list.append((rel_err_nuc, rel_err_L12))
        loss_list.append((loss_nuc, loss_L12))
        print(("id:{}. Current relative error: {}, log_losses: {},"
               "New test of type {} samples. Dimension:{}. Sparsity:{}. Iteration: {}").format(id,
                                                                                               (rel_err_nuc, rel_err_L12),
                                                                                               (loss_nuc, loss_L12),
                                                                                               len(S), p, d, it))
        S.extend(triplets(Ktrue, X, step, noise=True))
        Ks.append((Khat_nuc, Khat_L12))
    q.put({'pulls':len(S), 'Ks':Ks, 'n':n, 'd':d, 'p':p, 'start':5000, 'step':step, 'X':X, 
           'rel_err_list':rel_err_list, 'loss_list':loss_list})
    return len(S) 

def test_dim(n, d, p, step, avg=3, acc=0.1):
    seed = np.random.randint(1000)
    pulls = Parallel(n_jobs=min(8, len(d*avg)))\
            (delayed(estimate_pulls)(n, d[int(i/avg)], p[int(i/avg)], seed+i, q, step=step[int(i/avg)], acc=acc) for i in range(avg*len(d)))
    pulls = np.mean(np.array(pulls).reshape(int(len(d)/avg), avg), axis=1).tolist()
    return pulls

def writer_process(q):
    def writer(q):
        stream = io.open('test_dump','wb', buffering=0)
        while True:
            data = q.get()
            stream.write(msgpack.packb(data))
    p = mp.Process(target=writer, args=(q, ))
    p.daemon = True
    p.start()
    #p.join()

def reader_process(filename):
    stream = io.open(filename,'rb')
    unpacker = msgpack.Unpacker()
    lines = 1
    data = []
    while True:
        buf = stream.read(1024)
        if not buf:
            break
        unpacker.feed(buf)
        for o in unpacker:
            lines +=1
            data.append(o)
    return data

def get_data(stream_name):
    a = reader_process(stream_name)
    print(len(a))
    pulls, dims, log_loss = [], [], []
    for x in a:
        pulls.append(x['pulls'])
        dims.append(x['d'])
        log_loss.append(x['pred_err_list'])
    
    with open(stream_name + '_striped.json', 'wb') as outfile:
        json.dump({'pulls':pulls, 'dims':dims}, outfile)
    return pulls, dims

def plots():
    data = reader_process('risk_metric_p20_d6-20_n50')
    recovery_err_nuc = defaultdict(list)
    recovery_err_L12 = defaultdict(list)

    norm_nuc = defaultdict(list)
    norm_L12 = defaultdict(list)
    start = 5000
    for run in data:
        keys = ['rel_err_list', 'n', 'd', 'p', 'step']
        risks, n, d, p, step = [run[k] for k in keys]
        tries_risk_nuc = 5000
        tries_risk_L12 = 5000
        for r in risks:
            if r[0] > .003:
                tries_risk_nuc += step
            if r[1] > .003:
                tries_risk_L12 += step
        print 'd', d, tries_risk_nuc, tries_risk_L12
        norm_nuc[d].append(tries_risk_nuc)        
        norm_L12[d].append(tries_risk_L12)

        # Samples to recovery error
        recovery_samples_nuc = 5000
        recovery_samples_L12 = 5000
        Ktrue = np.eye(p)
        for i in range(d, p):
            Ktrue[i, i] = 0

        for K in run['Ks'][1:]:
            nuc_recovery = np.linalg.norm(Ktrue-K[0],'fro')**2/np.linalg.norm(Ktrue,'fro')**2
            L12_recovery = np.linalg.norm(Ktrue-K[1],'fro')**2/np.linalg.norm(Ktrue,'fro')**2
            if nuc_recovery > .10:
                recovery_samples_nuc += step
            if L12_recovery > .10:
                recovery_samples_L12 += step

        recovery_err_nuc[d].append(recovery_samples_nuc)
        recovery_err_L12[d].append(recovery_samples_L12)
    x = sorted(norm_nuc.keys())
    print x
    plt.figure(1)
    plt.subplot(211)
    plt.title('Samples to get excess_risk < .003')
    plt.errorbar(x, [np.mean(norm_nuc[d]) for d in x], yerr=[np.std(norm_nuc[d]) for d in x], color='red')
    plt.errorbar(x, [np.mean(norm_L12[d]) for d in x], yerr=[np.std(norm_L12[d]) for d in x])
    plt.subplot(212)
    plt.title('Samples to get generalization error < .01')
    plt.errorbar(x, [np.mean(recovery_err_nuc[d]) for d in x], yerr=[np.std(recovery_err_nuc[d]) for d in x], color='red')
    plt.errorbar(x, [np.mean(recovery_err_L12[d]) for d in x], yerr=[np.std(recovery_err_L12[d]) for d in x])

    plt.show()

if __name__ == '__main__':
    compute = eval(sys.argv[1])
    if compute:
        q = mp.Manager().Queue()
        writer_process(q)
        d = [10]#, 8, 10, 12, 14, 16, 18, 20]
        step = [2000]*len(d)                                 
        p = [20]*len(d)
        n = 30
        acc = 0.001       # accuracy relative to Xtrue to stop at
        avg = 4          # number of runs to average over
        pulls = test_dim(n, d, p, step, avg=avg, acc=acc)

        # q = mp.Manager().Queue()
        # writer_process(q)
        # d = [5, 10, 15, 20, 25 30, 35, 40, 45]
        # step = [50]*len(d)                                 
        # p = [50]*len(d)
        # n = 100
        # acc = 0.1       # accuracy relative to Xtrue to stop at
        # avg = 4          # number of runs to average over
        # pulls = test_dim(n, d, p, step, avg=avg, acc=acc)
    else:
        plots()

