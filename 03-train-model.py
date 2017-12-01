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



model = ftrl.FtrlProximal(alpha=0.1, beta=1, l1=75, l2=25)


for i in tqdm(range(30)):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    print(i + 1, auc)


X = sp.vstack([X_train, X_val])
y = np.concatenate([y_train, y_val])

model_full = ftrl.FtrlProximal(alpha=0.1, beta=1, l1=75, l2=25)
model_full.fit(X, y, num_passes=22)


# Apply the model

shift = 1.1875
scale = 850100

def shifted_scaled_sigmoid(x, shift=0, scale=1):
    s = 1 / (1 + np.exp(-x + shift))
    return (s * scale).round(2)


it_test = u.read_grouped('data/criteo_test_release.txt.gz')


f_out = open('pred_ftrl.txt', 'w')

for gid, group in tqdm(it_test, total=7087738):
    cols = []
    vals = []

    for line in group:
        cols.append(line.idx)
        vals.append(line.val)

    X_val = u.to_csr(cols, vals)

    pred = model_full.predict(X_val)
    pred = shifted_scaled_sigmoid(pred, shift, scale)

    m = pred.argmax()
    pred[m] = pred[m] + 15

    pred_str = u.to_prediction_str(gid, pred)
    
    f_out.write(pred_str)
    f_out.write('\n')

f_out.flush()
f_out.close()