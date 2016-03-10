import sys
#sys.path.append('/anaconda/lib/python2.7')
from numba.decorators import autojit
import numpy as np
import os, gzip
import hickle

def H1(data, block_size, step, counts, ent):

    entropy = 0.0
    #counts = np.zeros(17,dtype=np.float)
    for i in xrange(block_size):
        counts[data[i]] += 1

    for i in xrange(counts.shape[0]):
        if counts[i] > 0:
            entropy += - counts[i] / block_size * \
                np.log2(counts[i] / block_size)

    #cb0 = counts[np.where(counts>0)]

    #entropy = np.sum(-cb0/block_size * np.log2(cb0/block_size))
    ent[0] = entropy

    for i in xrange(1, data.shape[0] - block_size):

        dec = counts[data[i - 1]]
        inc = counts[data[i + block_size - 1]]

        counts[data[i - 1]] -= 1
        counts[data[i + block_size - 1]] += 1

        entropy -= -dec / block_size * np.log2(dec / block_size)
        if dec > 1:
            entropy += -(dec - 1) / block_size * \
                np.log2((dec - 1) / block_size)

        if inc > 0:
            entropy -= -inc / block_size * np.log2(inc / block_size)

        entropy += - (inc + 1) / block_size * np.log2((inc + 1) / block_size)

        if i % step == 0:
            ent[i / step] = (entropy)


H_numba = autojit(H1, nopython=True)


def get_entropy_features(byte_data):

    corr = {str(key): key for key in range(10)}
    corrl = {'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, '?': 16}
    corr.update(corrl)

    block_size = 10000
    step = 100
    text = byte_data
    #name = filename.split('/')[-1].split('.')[0]

    #with gzip.open(filename, 'r') as f:
    #    text = f.read()

    lines = ''.join(byte_data).split('\r\n')

    t = []
    for l in lines:
        elems = l.split(' ')
        t.extend(elems[1:])
    t = ''.join(t)

    chararray = np.array([corr[x] for x in t])

    counts = np.zeros(17, dtype=np.float)
    ent = np.zeros((chararray.shape[0] - block_size) / step + 1)
    H_numba(chararray, block_size, step, counts, ent)

    #return [name, ent]
    return ent


def get_qdiffs(data, r):
    q = []
    prev = 0
    for alpha in r:
        t = float((data < alpha).sum()) / data.shape[0]
        q.append(t - prev)
        prev = t
    return q


def get_percentiles(data, num, diffs=False):
    d = np.sort(data)

    step = np.floor(float(d.shape[0]) / num) - 1

    p = [d[step * i] for i in range(num)]
    if diffs:
        return np.ediff1d(p, to_begin=p[0])

    return p


def get_feats(data_raw):
    #data_raw = np.array(data_raw)
    num_blocks = 4
    feats = []
    for i in xrange(len(data_raw)):

        f = []

        # feats quatiles
        q_diffs = get_qdiffs(data_raw[i], np.arange(0.2, 4.4, 0.2))
        f.extend(q_diffs)

        # feats stats
        f.extend([np.mean(data_raw[i]), np.var(data_raw[i]), np.median(data_raw[i]), np.max(
            data_raw[i]), np.min(data_raw[i]), np.max(data_raw[i]) - np.min(data_raw[i])])

        # diffs quatiles
        d_f = np.diff(data_raw[i])
        q_diffs = get_qdiffs(d_f, np.arange(-0.1, 0.11, 0.01))
        f.extend(q_diffs)
        # diffs stats
        f.extend([np.mean(d_f), np.var(d_f), np.median(d_f), np.max(
            d_f), np.min(d_f), np.max(d_f) - np.min(d_f)])

        block_size = data_raw[i].shape[0] / num_blocks

        for j in range(num_blocks):
            bl = data_raw[i][j * block_size:(j + 1) * block_size]

            q_diffs = get_qdiffs(bl, np.arange(0.2, 4.4, 0.2))
            f.extend(q_diffs)

            # feats stats
            f.extend([np.mean(bl), np.var(bl), np.median(bl), np.max(
                bl), np.min(bl), np.max(bl) - np.min(bl)])

        p = get_percentiles(data_raw[i], 20, diffs=False)
        f.extend(p)

        p_diffs = get_percentiles(data_raw[i], 20, diffs=True)
        f.extend(p_diffs)

        feats.append(f)
    return feats


def dump_names(ent_feats_dir):
    st = ['mean','var','median','max','min','max-min']

    n = []
    n.extend( ['ent_q_diffs_' + str(x) for x in range(21) ])
    n.extend( ['ent_q_diffs_' + x for x in st])

    n.extend( ['ent_q_diff_diffs_' + str(x) for x in range(21) ])
    n.extend( ['ent_q_diff_diffs_' + x for x in st])

    for i in range(4):
        n.extend( ['ent_q_diff_block_' + str(i) + '_' + str(x) for x in range(21) ])
        n.extend( ['ent_q_diff_diffs_'+ str(i) + '_' + x for x in st])

    n.extend( ['ent_p_' + str(x) for x in range(20) ])
    n.extend( ['ent_p_diffs_' + str(x) for x in range(20) ])

    hickle.dump(n,os.path.join(ent_feats_dir,'ent_feats_names'))