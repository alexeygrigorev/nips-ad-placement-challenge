# coding: utf-8

# the ftrl package is from
# https://github.com/alexeygrigorev/libftrl-python

import ftrl

import pickle
import gzip

import pandas as pd
import numpy as np
import scipy.sparse as sp

from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import competition_utils as u


# Train the model

X_train = sp.load_npz('tmp/X_train_sparse.npz')
X_val = sp.load_npz('tmp/X_val_sparse.npz')

y_train = np.load('tmp/y_train.npy', ).astype(np.float32)
y_val = np.load('tmp/y_val.npy', ).astype(np.float32)

X = sp.vstack([X_train, X_val])
y = np.concatenate([y_train, y_val])


for i in tqdm(range(10)):
    model = ftrl.FtrlProximal(alpha=0.1, beta=1, l1=75, l2=25)
    model.fit(X, y, num_passes=22)
    model.save_model('tmp/model_%i.bin' % i)

