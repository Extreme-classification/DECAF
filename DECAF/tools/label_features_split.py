import xclib.data.data_utils as du
import sys
import os
import numpy as np
import _pickle as pickle
import json


def main():
    data_dir = sys.argv[1]
    trn_x = du.read_sparse_file(os.path.join(data_dir, sys.argv[2]))
    trn_y = du.read_sparse_file(os.path.join(data_dir, sys.argv[3]))
    yft_x = du.read_sparse_file(os.path.join(data_dir, sys.argv[4]))
    tmp_mdata = sys.argv[5]
    assert trn_x.shape[0] == trn_y.shape[0], "Number of instances must be same in features and labels"
    num_labels = trn_y.shape[1]
    valid_trn_x = np.where(trn_x.getnnz(axis=1) > 0)[0]
    valid_trn_y = np.where(trn_y.getnnz(axis=1) > 0)[0]
    valid_idx = np.intersect1d(valid_trn_x, valid_trn_y)
    trn_x = trn_x[valid_idx]
    trn_y = trn_y[valid_idx]
    features = np.where(trn_x.getnnz(axis=0) > 0)[0]
    labels = np.where(trn_y.getnnz(axis=0) > 0)[0]
    v_lbs_wrds = np.where(yft_x[labels].getnnz(axis=0) > 0)[0]
    union_fts = np.union1d(v_lbs_wrds, features)
    path = os.path.join(tmp_mdata, 'features_split.txt')
    np.savetxt(path, union_fts, fmt='%d')
    path = os.path.join(tmp_mdata, 'labels_split.txt')
    np.savetxt(path, labels, fmt='%d')
    path = os.path.join(tmp_mdata, 'v_lbs_fts_split.txt')
    np.savetxt(path, union_fts, fmt='%d')
    params = "{},{},{},{}".format(union_fts.size, num_labels,
                                  labels.size, union_fts.size)
    print(params)
    stats_obj = {'header': 'num_features,num_labels,valid_num_labels, valid_num_features'}
    stats_obj['all'] = params
    json.dump(stats_obj, open(os.path.join(
        tmp_mdata, "split_stats.json"), 'w'), indent=4)


if __name__ == '__main__':
    main()
