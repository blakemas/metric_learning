from __future__ import division, print_function
import json, sys

import numpy as np 
from scipy.io import loadmat
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

from utilsMetric import *

def getData(file_triplets, file_features):
    S = loadmat(file_triplets)['triplets']
    print(S.shape)
    Strain, Stest = train_test_split(S, test_size=0.4, random_state=42)

    with open(file_features, 'r') as json_data:
        data = json.load(json_data)

    X = data['X']
    features = data['features_list']
    return Strain, Stest, X, features

def learn(Strain, Stest, X, lams):
    Stest_1 = Stest[:int(len(Stest)/2)]     # first test set
    Stest_2 = Stest[int(len(Stest)/2):]

    best_loss = float('inf')
    lam_best = None
    d = 50              # not actually used. Just for compatibility
    training_losses = []

    for i, lam in enumerate(lams):
        print('Run ', i + 1, ' of ', len(lams))
        K, emp_loss, log_loss = computeKernel(X, Strain.tolist(), d, lam,
                                              regularization='norm_L12', 
                                              maxits=250,
                                              alpha=300.,
                                              c1 = 1e-7, 
                                              epsilon=1e-8, 
                                              verbose=True)
        training_losses.append(emp_loss[-1])
        test_emp_loss, new_loss = getLoss(K, X, Stest_1)
        if new_loss <= best_loss:
            best_loss = new_loss
            lam_best = lam
        print('Test_loss: ', new_loss, ' Best loss: ', best_loss, 'emp_loss: ', test_emp_loss)

    ### Use best lambda to retrain with training data and 1st test set
    Khat, train_emp_loss, train_log_loss = computeKernel(X, np.vstack([Strain, Stest_1]).tolist(), d, lam_best,
                                              regularization='norm_L12', 
                                              maxits=750,
                                              alpha=300,
                                              c1 = 1e-7, 
                                              epsilon=1e-10, 
                                              verbose=True)
    test_emp_loss, test_log_loss = getLoss(Khat, X, Stest_2)
    print('Empirical loss: ({}, {}), Logistic loss: ({}, {}), lambda: {}' \
                                                                .format(train_emp_loss[-1], test_emp_loss,
                                                                train_log_loss[-1], test_log_loss,
                                                                lam_best))
    print(training_losses)
    return Khat, train_log_loss, test_log_loss

def plot_Khat(K):
    cmap = 'plasma'
    plt.imshow(K, interpolation='none', cmap=cmap)
    plt.axis('off')
    return

if __name__ == '__main__':
    # p = 50
    # n = 50
    # # lams = [np.sqrt(5*i)*p for i in range(1, 22, 2)]
    # lams = [1750]
    Strain, Stest, X, features = getData('lewis_triplets_HWL.mat','lewis_features.json')
    # X = np.array(X)

    # # center and normalize X
    # X = X / np.sqrt(np.sum(X * X, axis=1)).reshape(n, 1)
    # V = np.eye(n) - 1/n * np.ones((n,n))
    # X = X.dot(V) 
    # Khat, train_log_loss, test_log_loss = learn(Strain, Stest, X, lams)
    Khat = loadmat('Khat_cvx.mat')['Khat']
    plot_Khat(Khat)
    plt.savefig('khat_cvx.png', dpi=600)
    plt.show()

    top_feats = np.argsort(np.sqrt(np.sum(Khat * Khat, axis=1)))[::-1][:20]     # top 10 features by weight
    print(np.array(features)[top_feats])

    for f in top_feats:
        print(np.linalg.norm(Khat[f, :]))

    evals, _ = np.linalg.eig(Khat)
    plt.plot(evals)
    plt.show()
    print(evals)

