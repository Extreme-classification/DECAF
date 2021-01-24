import scipy.sparse as sp
import numpy as np
import sys
import os

def _score_compute(score, factor=1):
    score.__dict__['data'] = np.exp(score.__dict__['data'])/factor
    return score

if __name__ == '__main__':
    result_dir = sys.argv[1]
    num_trees = int(sys.argv[2])
    fname = sys.argv[3]
    file_name = os.path.join(result_dir, str(num_trees), fname)
    score_mat = _score_compute(sp.load_npz(file_name), num_trees)
    for i in range(1, num_trees):
        file_name = os.path.join(result_dir, str(num_trees-i), fname)
        score_mat+= _score_compute(sp.load_npz(file_name), num_trees)
    fname = fname.replace(".npz", "_num_trees={}.npz".format(num_trees))
    sp.save_npz(os.path.join(result_dir, fname), score_mat)
