# coding: utf-8

# the ftrl package is from
# https://github.com/alexeygrigorev/libftrl-python

import ftrl

from glob import glob

import numpy as np
import scipy.sparse as sp

from tqdm import tqdm

import competition_utils as u


# apply the model

model_files = sorted(glob("tmp/model*.bin"))
print('loading the models from %s...' % model_files)

models = []
for mf in model_files:
    model = ftrl.load_model(mf)
    models.append(model)


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

    preds = []
    for model in models:
        pred = model.predict(X_val)
        pred = shifted_scaled_sigmoid(pred, shift, scale)
        preds.append(pred)

    pred = np.mean(preds, axis=0)
    m = pred.argmax()
    pred[m] = pred[m] + 15

    pred_str = u.to_prediction_str(gid, pred)

    f_out.write(pred_str)
    f_out.write('\n')

f_out.flush()
f_out.close()
