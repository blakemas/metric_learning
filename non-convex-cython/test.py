import numpy as np 
from scipy.linalg import orth
import matplotlib.pyplot as plt 

from utilsMetric import *

def setup_problem(n, d, p, samples, noise=False, seed=42):
    '''
    Generate an example of p points with a rank d metric.
    Gather 'samples' number of triplet queries.
    '''
    np.random.seed(seed)
    U = np.sqrt(p/np.sqrt(d))*orth(np.random.randn(p, d))
    K = np.dot(U, U.T)
    X = np.random.randn(n, p) * 1/p**.5
    S = triplets(K, X, samples, noise=True)
    return U, X, S 

def estimate_U(X, S, d, lam, maxits=100, alpha=4.):
    Uhat, emp_loss, hinge_loss = computeMap(X, S, d, lam, 
                                              regularization='fro',
                                              c1=.1, 
                                              rho=0.7, 
                                              maxits=maxits,
                                              alpha=alpha, 
                                              epsilon=0, 
                                              verbose=True,
                                              Ustart=None)
    return Uhat, emp_loss, hinge_loss

if __name__ == '__main__':
    # experiment parameters
    n, d, p = 100, 3, 10
    samples = 2000

    # setup experiment
    Utrue, X, S = setup_problem(n, d, p, samples)

    # estimate U
    lam, maxits, alpha = 1e-6, 50, 2
    Uhat, emp_loss, hinge_loss = estimate_U(X, S, d, lam, maxits=maxits, alpha=alpha)

    # error performance
    final_emp, final_hinge = emp_loss[-1], hinge_loss[-1]
    Ktrue, Khat = Utrue @ Utrue.T, Uhat @ Uhat.T 
    psd_rel_err = np.linalg.norm(Ktrue - Khat, ord='fro')/np.linalg.norm(Ktrue, ord='fro')
    print('Final hinge loss: {}, Final 0/1 loss: {}, Final relative Error: {}'\
                    .format(final_hinge, final_emp, psd_rel_err))

    # plot error curves
    plt.figure(1)
    plt.plot(emp_loss, 'r', label='0/1 loss')
    plt.plot(hinge_loss, 'b', label='hinge loss')
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.title('Loss per iteration')
    plt.show()
