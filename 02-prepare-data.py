# coding: utf-8

import gzip
import itertools

import pandas as pd
import numpy as np
import scipy.sparse as sp

from tqdm import tqdm

import competition_utils as u


it0 = u.read_data('data/train_0.txt', skip_unlabelel=True)
it1 = u.read_data('data/train_1.txt', skip_unlabelel=True)
it2 = u.read_data('data/train_2.txt', skip_unlabelel=True)

it_train = itertools.chain(it0, it1, it2)

df_train = []

for line in tqdm(it_train):
    df_train.append(line)

df_train = pd.DataFrame(df_train)


it_val = read_train('data/train_3.txt')

df_val = []

for line in tqdm(it_val):
    df_val.append(line)

df_val = pd.DataFrame(df_val)


X_train = u.to_csr(list(df_train.idx), list(df_train.val))
X_val = u.to_csr(list(df_val.idx), list(df_val.val))

sp.save_npz('tmp/X_train_sparse.npz', X_train, compressed=False)
sp.save_npz('tmp/X_val_sparse.npz', X_val, compressed=False)


y_train = df_train.label.values.astype('uint8')
y_val = df_val.label.values.astype('uint8')

np.save('tmp/y_train.npy', y_train)
np.save('tmp/y_val.npy', y_val)