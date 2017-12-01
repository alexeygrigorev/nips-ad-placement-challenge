# coding: utf-8

import gzip

from itertools import groupby
from collections import namedtuple

import numpy as np
import scipy.sparse as sp


Line = namedtuple('Line', ['id', 'f0', 'f1', 'idx', 'val'])
LabeledLine = namedtuple('LabeledLine', ['id', 'f0', 'f1', 'idx', 'val', 'propensity', 'label'])

def parse_features(s):
    split = s.split(' ')
    f0 = split[0]
    assert f0.startswith('0:')
    f0 = int(f0[2:])

    f1 = split[1]
    assert f1.startswith('1:')
    f1 = int(f1[2:])

    idx = []
    values = []
    
    for fv in split[2:]:
        f, v = fv.split(':')
        idx.append(int(f) - 2)
        values.append(int(v))

    return f0, f1, idx, values

def read_data(fname, skip_unlabelel=True):
    if fname.endswith('.gz'):
        fin = gzip.open(fname, 'r')
        f = map(bytes.decode, fin)
    else:
        fin = open(fname, 'r')
        f = fin

    for line in f:
        split = line.split('|')
        id = int(split[0].strip())

        if len(split) == 4:
            l = split[1]
            assert l.startswith('l')

            l = l.lstrip('l ').strip()
            if l == '0.999':
                label = 0
            elif l == '0.001':
                label = 1
            else:
                raise Exception('ololo')

            p = split[2]
            assert p.startswith('p')
            p = p.lstrip('p ').strip()
            propensity = float(p)

            features = split[3].lstrip('f ').strip()
            f0, f1, idx, val = parse_features(features)
            idx = np.array(idx, dtype=np.uint32)
            val = np.array(val, dtype=np.uint8)
            yield LabeledLine(id, f0, f1, idx, val, propensity, label)

        elif len(split) == 2 and not skip_unlabelel:
            features = split[1].lstrip('f ').strip()

            f0, f1, idx, val = parse_features(features)
            idx = np.array(idx, dtype=np.uint32)
            val = np.array(val, dtype=np.uint8)
            yield Line(id, f0, f1, idx, val)

    fin.close()


def read_grouped(fname):
    it = read_data(fname, skip_unlabelel=False)
    groups = groupby(it, key=lambda x: x.id)
    for id, group in groups:
        yield id, group


def to_csr(cols, vals, shape=74000):
    lens = [len(c) for c in cols]
    intptr = np.zeros((len(cols) + 1), dtype='int32')
    intptr[1:] = lens
    intptr = intptr.cumsum()

    columns = np.concatenate(cols).astype('int32')
    values = np.concatenate(vals).astype('uint8')

    return sp.csr_matrix((values, columns, intptr), shape=(len(cols), shape))

def to_prediction_str(id, preds):
    res = ['%d:%0.2f' % (i, p) for (i, p) in enumerate(preds)]
    return '%d;%s' % (id, ','.join(res))