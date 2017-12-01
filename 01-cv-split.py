# coding: utf-8

import gzip
from tqdm importtqdm

train_input = gzip.open('data/criteo_train.txt.gz', 'r')

files = [
    open('data/train_0.txt', 'w'),
    open('data/train_1.txt', 'w'),
    open('data/train_2.txt', 'w'),
    open('data/train_3.txt', 'w'),
]

for line in tqdm(train_input):
    line = line.decode()
    split = line.split('|')
    id = int(split[0].strip())
    fold = hash(id) % 4
    files[fold].write(line)

for f in files:
    f.flush()
    f.close()

