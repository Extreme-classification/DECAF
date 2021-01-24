from scipy.sparse import lil_matrix, csr_matrix, hstack, load_npz, save_npz
from xclib.data import data_utils
from scipy.io import loadmat, savemat
import pickle
import numpy as np
import os
import sys

class CombineResults(object):
    def __init__(self,ftype):
        self.label_mapping = [] # Mapping from new to original in each decile
        self.ftype = ftype

    def read_predictions(self, fname):
        if self.ftype == 'mat':
            return loadmat(fname)['predicted_labels']
        elif self.ftype == 'txt':
            return data_utils.read_sparse_file(fname)
        elif self.ftype == 'npz':
            return load_npz(fname)

    def write_predictions(self,file,fname):
        print("Saving at %s"%(fname))
        if self.ftype == 'mat':
            savemat(fname,{'predicted_labels':file})
        elif self.ftype == 'txt':
            data_utils.write_sparse_file(file,fname)
        elif self.ftype == 'npz':
            save_npz(fname, file)

    def read_mapping(self, fname, _set):
        self.label_mapping = np.loadtxt(fname, dtype=np.int32)

    def map_to_original(self, mat, label_map, n_cols):
        n_rows = mat.shape[0]
        row_idx, col_idx = mat.nonzero()
        vals = np.array(mat[row_idx, col_idx]).squeeze()
        col_indices = list(map(lambda x:label_map[x], col_idx))
        return csr_matrix((vals, (row_idx, np.array(col_indices))), shape=(n_rows, n_cols))
            
    def combine(self, num_labels, fname_predictions, fname_mapping, _set='test', ftype='npz'):
        self.read_mapping(fname_mapping, _set)
        pred = self.read_predictions(fname_predictions)
        pred_mapped = self.map_to_original(pred.tocsr(), self.label_mapping, num_labels)
        return pred_mapped


def main():
    result_dir = sys.argv[1]
    suffix_predictions = sys.argv[2]
    mapping_dir = sys.argv[3]
    num_labels = int(sys.argv[4])
    cr = CombineResults('npz')
    fname_predictions = os.path.join(result_dir, "XC", suffix_predictions)
    fname_mapping = os.path.join(mapping_dir, "labels_split.txt")
    predicted_labels = cr.combine(num_labels, fname_predictions, fname_mapping)
    cr.write_predictions(predicted_labels, os.path.join(result_dir, suffix_predictions))
    print("Results combined!")



if __name__ == "__main__":
    main()
