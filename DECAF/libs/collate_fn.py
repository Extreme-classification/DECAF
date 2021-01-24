import torch
import numpy as np
import scipy.sparse as sp


def _block_sparse_matrix(label_words):
    data = torch.FloatTensor(label_words.data)
    idx = torch.LongTensor(np.vstack(label_words.nonzero()))
    shape = torch.Size(label_words.shape)
    return idx, data, shape


def _create_sparse_mat(cols, data, shape):
    rows = list(map(lambda x: [x[0]]*len(x[1]), enumerate(cols)))
    cols = np.concatenate(cols)
    rows = np.concatenate(rows)
    data = np.concatenate(data)
    return sp.coo_matrix((data, (rows, cols)), shape=shape).tocsc()


def _return_padded_from_list(Var, padding_val,
                             dtype=torch.FloatTensor):
    return torch.nn.utils.rnn.pad_sequence(
        list(map(lambda x: torch.from_numpy(
            np.asarray(x)).type(dtype), Var)),
        batch_first=True, padding_value=padding_val)


def _return_padded_batch(Var, idx, key, padding_val,
                         dtype=torch.FloatTensor):
    return torch.nn.utils.rnn.pad_sequence(
        list(map(lambda x: torch.from_numpy(
            x[idx][key]).type(dtype), Var)),
        batch_first=True, padding_value=padding_val)


class construct_collate_fn:
    """
    Collate function for the dataset of DECAF
    """

    def __init__(self, feature_type, use_shortlist=False,
                 num_partitions=1, sparse_label_fts=None,
                 freeze=False, padding_idx=0, num_labels=-1):
        self.num_partitions = num_partitions
        self.padding_idx = padding_idx
        self.sparse_label_fts = sparse_label_fts
        if self.sparse_label_fts is not None:
            self.sparse_label_fts = self.sparse_label_fts
        self.freeze = freeze
        self.num_labels = num_labels
        self.use_shortlist = use_shortlist
        self._get_docs(feature_type)
        self._batcher(sparse_label_fts, use_shortlist)

    def __call__(self, batch):
        return self.batcher(batch)

    def _batcher(self, sparse_label_fts, use_shortlist):
        if sparse_label_fts is not None:
            if use_shortlist:
                self.sparse_label_fts = self.sparse_label_fts.tocsr()
                self.batcher = self.collate_fn_shorty_lbf
            else:
                self.batcher = self.collate_fn_full_lbf
        else:
            if use_shortlist:
                self.batcher = self.collate_fn_shorty
            else:
                self.batcher = self.collate_fn_full

    def _get_docs(self, feature_type):
        """
            Collates document features
        """
        if feature_type == 'dense':
            self.collate_docs = self.collate_fn_docs_dense
        elif feature_type == 'sparse':
            self.collate_docs = self.collate_fn_docs_sparse
        else:
            print("Kuch bhi")

    def collate_fn_docs_dense(self, batch, batch_data):
        """
            Dense document feature collator
        """
        batch_data['X'] = torch.stack(list(
            map(lambda x: torch.from_numpy(x[0]), batch)
        ), 0).type(torch.FloatTensor)
        batch_data['batch_size'] = len(batch)
        batch_data['idx'] = np.arange(
            batch_data['batch_size']
        ).reshape(-1, 1)
        batch_data['is_sparse'] = False

    def collate_fn_docs_sparse(self, batch, batch_data):
        """
            BoW document feature collator
        """
        batch_data['X'] = _return_padded_batch(batch, 0, 0, self.padding_idx,
                                               dtype=torch.LongTensor)
        batch_data['X_w'] = _return_padded_batch(batch, 0, 1, 0,
                                                 dtype=torch.FloatTensor)
        batch_data['batch_size'] = len(batch)
        batch_data['idx'] = np.arange(batch_data['batch_size']).reshape(-1, 1)
        batch_data['is_sparse'] = True

    def collate_fn_full_lbf(self, batch):
        """
            OvA collation for the labels
        """
        batch_data = {}
        self.collate_docs(batch, batch_data)
        batch_data['Y'] = torch.from_numpy(np.vstack(
            list(map(lambda x: x[1], batch))
        )).type(torch.FloatTensor)
        return batch_data

    def collate_fn_shorty_lbf(self, batch):
        """
            Collation shortlist of labels
        """
        batch_data = {}
        self.collate_docs(batch, batch_data)
        if self.num_partitions > 1:
            batch_data['Y_m'] = torch.stack(
                list(map(lambda x: torch.LongTensor(x[1][3]), batch)), 0)
        Y_s = _return_padded_batch(batch, 1, 0, self.num_labels,
                                   dtype=torch.LongTensor)
        batch_data['Y'] = _return_padded_batch(batch, 1, 1, 0,
                                               dtype=torch.FloatTensor)
        v_lbl_idx = torch.unique(Y_s)
        if not self.freeze:
            """
                Building sparse matrix for shortlist only if 
                embeddings are set to trainable
            """
            valid_labels = v_lbl_idx.numpy()
            v_lbl_wrd = self.sparse_label_fts[valid_labels].tocsc()
            v_lbl_fts = np.where(v_lbl_wrd.getnnz(axis=0) > 0)[0]
            v_lbl_wrd = _block_sparse_matrix(v_lbl_wrd[:, v_lbl_fts])
            v_lbl_fts = torch.from_numpy(v_lbl_fts).type(torch.LongTensor)
            batch_data['v_lbl_fts'] = v_lbl_fts
            batch_data['v_lbl_wrd'] = v_lbl_wrd

        Y_r = torch.zeros(self.num_labels + 1, dtype=torch.long)
        Y_r[v_lbl_idx] = torch.arange(len(v_lbl_idx), dtype=torch.long)
        batch_data['Y_lbl_rmp'] = Y_r[Y_s]
        batch_data['v_lbl_idx'] = v_lbl_idx
        return batch_data

    def collate_fn_shorty(self, batch):
        """
            Used for module 4 prediction only
        """
        batch_data = {}
        self.collate_docs(batch, batch_data)
        batch_data['Y_s'] = _return_padded_batch(batch, 1, 0, self.num_labels,
                                                 dtype=torch.LongTensor)
        batch_data['Y_d'] = _return_padded_batch(batch, 1, 2, -np.inf,
                                                 dtype=torch.FloatTensor)
        return batch_data

    def collate_fn_full(self, batch):
        """
            Used for module 1,2 prediction only
        """
        batch_data = {}
        self.collate_docs(batch, batch_data)
        if self.num_partitions > 1:
            batch_data['Y'] = list(map(lambda idx: torch.stack(
                list(map(lambda x: torch.from_numpy(x[1][idx]), batch)), 0),
                range(self.num_partitions)))
        else:
            batch_data['Y'] = torch.from_numpy(np.vstack(
                list(map(lambda x: x[1], batch))
            )).type(torch.FloatTensor)
        return batch_data
