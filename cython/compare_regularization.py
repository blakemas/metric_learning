from __future__ import division, print_function

import json

import numpy as np 
from joblib import Parallel, delayed
from multiprocessing import cpu_count

import utilsMetric as utils

def crossValidate(S, X, lams, d, regularization='L12', max_its=200, epsilon=1e-6, rho=0.5)