from torch.utils.data import DataLoader
import torch

import xclib.evaluation.xc_metrics as xc
from xclib.utils.sparse import retain_topk
from scipy.sparse import lil_matrix, issparse

from .collate_fn import construct_collate_fn
from .utils import load_overlap
from .dataset import construct_dataset
from .tracking import Tracking

import numpy as np
import logging
import time
import sys
import os


class ModelDECAFBase(object):
    """
        Base class for DECAF
        Implements base class for all the modules.
        Arguments
            -----------
            params: NameSpace
                parameter of the model
            net: nn.Module
                Network for model training
            criterion: nn.Loss
                Loss function for calculating back propagation
            optimizer: nn.optim
                Optimzer for paramer updates
    """

    def __init__(self, params, net, criterion, optimizer, *args, **kwargs):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_rate = 0.001
        self.model_dir = params.model_dir
        self.last_epoch = 0
        self.dlr_step = 0
        self.dlr_factor = params.dlr_factor
        self.progress_step = 500
        self.model_fname = params.model_fname
        self.logger = self.get_logger(name=self.model_fname)
        self.filter_docs, self.filter_lbls = load_overlap(
            os.path.join(params.data_dir, params.dataset),
            params.label_indices, filter_label_file=params.filter_labels)
        self.lrs = params.depth_lrs
        self.call_back = params.call_backs
        self.beam = params.beam_size
        self.batch_size = params.batch_sizes
        self.dlr_steps = params.dlr_steps
        self.tracking = Tracking()

    def _create_dataset(self, data_dir, fname_features, fname_labels=None,
                        data=None, mode='predict', normalize_features=True,
                        normalize_labels=False, feature_type="sparse",
                        keep_invalid=False, feature_indices=None,
                        label_indices=None, size_shortlist=None,
                        shortlist_method='static', classifier_type="DECAF"):
        """
            Create dataset as per given parameters

        """
        _dataset = construct_dataset(
            data_dir=data_dir,
            fname_features=fname_features, fname_labels=fname_labels,
            data=data, model_dir=self.model_dir, mode=mode,
            size_shortlist=size_shortlist,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            keep_invalid=keep_invalid, num_centroids=1,
            feature_type=feature_type,
            feature_indices=feature_indices,
            label_indices=label_indices,
            shortlist_method=shortlist_method,
            classifier_type=classifier_type)
        return _dataset

    def _create_dl(self, dataset, batch_size=128,
                   num_workers=4, shuffle=False,
                   mode='predict', use_shortlist=False,
                   sparse_label_fts=None):
        """
            Create data loader for given dataset
        """
        dt_loader = DataLoader(
            dataset, batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=construct_collate_fn(
                dataset.feature_type, use_shortlist,
                sparse_label_fts=sparse_label_fts,
                padding_idx=dataset.features.num_features,
                num_labels=dataset.num_labels),
            shuffle=shuffle)
        return dt_loader

    def _prep_ds(self, depth, dataset):
        """
            builds dataset [train|val|test] based on depth
            Arguments
                ----------
                depth: int
                    0: module 1 and 2
                    1: module 4
                dataset: Class object
            Returns
                ----------
                float: prediction time
        """
        print("learning for depth %d" % (depth))
        preds, pred_time = None, 0
        if depth > 0:
            dl = self._prep_dl(depth-1, dataset, mode='test')
            preds, _ = self._predict_depth(depth-1, dl)
            self._print_stats(dataset.labels.labels, preds)
        clusters = self.net.tree._get_cluster_depth(depth)
        dataset._prep_for_depth(depth, clusters)
        dataset.build_shortlist(self.net, preds)
        del preds
        return pred_time

    def _prep_dl(self, d, dataset, mode='test'):
        """
        Creates dataloader for a given dataset
        Argument:
            ----------
            d: int
                0: module 1 and 2
                1: module 4
            dataset: Class object
                dataset
            mode: str
                mode for the dataset
        Returns:
            ----------
            torch.utils.data.DataLoader: dataloader for the dataset
        """
        if mode == "train":
            return self._create_dl(dataset, batch_size=self.batch_size[d],
                                   num_workers=6, shuffle=True,
                                   mode=mode, use_shortlist=np.bool(d > 0),
                                   sparse_label_fts=self.net.label_words)
        else:
            return self._create_dl(dataset, batch_size=self.batch_size[d],
                                   num_workers=6, shuffle=False,
                                   mode=mode, use_shortlist=np.bool(d > 0))

    def transfer_to_devices(self):
        """
            Transfer network to devices
        """
        self.net.to()

    def _prep_for_depth(self, depth, train_ds, valid_ds):
        """
            Creates dataloader for train module 2 as well as module 4
            and uses model 3 to initliaze module 2 and 4
            Arguments
                ----------
                depth: int
                    0: for module 2
                    1: for module 4
                train_ds: Class Object
                    train dataset
                valid_ds: Class Object
                    Validation dataset
            Returns:
                ----------
                torch.utils.data.Dataloader: training dataloader
                torch.utils.data.Dataloader: validation dataloader
        """
        train_ds.mode = 'test'
        self._prep_ds(depth, train_ds)
        self._prep_ds(depth, valid_ds)
        self.net.cpu()
        self.net._prep_for_depth(depth)
        print(self.net)
        self.optimizer.learning_rate = self.lrs[depth]
        self.learning_rate = self.lrs[depth]
        self.optimizer.construct(self.net.depth_node, None)
        self.dlr_step = self.dlr_steps[depth]
        train_ds.mode = 'train'
        train_dl = self._prep_dl(depth, train_ds, mode='train')
        valid_dl = self._prep_dl(depth, valid_ds, mode='test')
        return train_dl, valid_dl

    def get_logger(self, name='DECAF', level=logging.INFO):
        """
            Return logging object!
        """
        logging.basicConfig(level=level, stream=sys.stdout)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        return logger

    def _compute_loss_one(self, _pred, _true):
        """
            Computes loss
            Arguments:
                _pred: torch.Tensor
                    prediction from model
                _true: torch.Tensor
                    ground truth
            Returns:
                torch.float: loss for the batch
        """
        device = _pred.get_device()
        if device == -1:
            device = "cpu"
        _true = _true.to(device)
        return self.criterion(_pred, _true).to(device)

    def _compute_loss(self, out_ans, batch_data):
        return self._compute_loss_one(out_ans, batch_data['Y'])

    def _step(self, data_loader):
        """
            Iterate over entrie dataset
            Arguments:
                data_loader: torch.utils.data.DataLoader
            Returns:
                float: Mean loss for each datapoint
        """
        self.net.cpu()
        self.net.train()
        self.transfer_to_devices()
        torch.set_grad_enabled(True)
        num_batches, mean_loss = len(data_loader), 0
        for batch_idx, batch_data in enumerate(data_loader):
            self.optimizer.zero_grad()
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data)
            loss = self._compute_loss(out_ans, batch_data)
            mean_loss += loss.item()*batch_size
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.progress_step == 0:
                self.logger.info(
                    "Training progress: [{}/{}]".format(
                        batch_idx, num_batches))
            del batch_data
        self.optimizer.zero_grad()
        self.net.cpu()
        return mean_loss / data_loader.dataset.num_instances

    def _print_stats(self, Y, mat, _filter=True):
        """
            Print statistics in terms of P@1,3,topk and nDCG@1,3,topk
        """
        _prec, _ndcg, _recall = self.evaluate(Y, mat, _filter)
        self.tracking.val_precision.append(_prec)
        self.tracking.val_precision.append(_ndcg)
        self.logger.info("R@{}: {}, R@{}: {}, R@{}: {}".format(
            1, _recall[0]*100, 3, _recall[2]*100,
            self.beam, _recall[self.beam-1]*100))
        self.logger.info("P@{}: {}, P@{}: {}, P@{}: {}".format(
            1, _prec[0]*100, 3, _prec[2]*100,
            self.beam, _prec[self.beam-1]*100))

    def validate(self, valid_loader, model_dir=None, epoch=None):
        """
            Wrapper for calculating Metrics for vaildation dataset
        """
        predicted_labels, prediction_time = self._predict_depth(
            self.net.depth, valid_loader)
        self.tracking.validation_time = self.tracking.validation_time \
            + prediction_time
        Y = valid_loader.dataset.labels.labels
        self._print_stats(Y, predicted_labels)

    def _predict_depth(self, depth, data_loader, **kwargs):
        """
        prediction function for module 1,2, and 4
        Arguments
            ----------
            depth: int
                0: for module 1 and 2
                1: for module 4
            data_loader: torch.utils.data.DataLoader
                data loader for the dataset to make prediction
        Returns
            ----------
            CSR matrix: Prediction matrix for the dataset
            float: prediction time for the module
        """
        self.net.train()
        self.net.cpu()
        self.net.eval()
        torch.set_grad_enabled(False)
        torch.cuda.empty_cache()
        self.transfer_to_devices()
        num_inst = data_loader.dataset.num_instances
        num_lbls = data_loader.dataset.num_labels
        num_batches = len(data_loader)
        pred_lbs = lil_matrix((num_inst, num_lbls + self.net.offset))
        self.logger.info("Preping the classifier")
        batch_pred_time, count = 0, 0
        for batch_idx, batch_data in enumerate(data_loader):
            time_begin = time.time()
            score = self.net._predict(batch_data)
            if depth > 0:
                index = batch_data['Y_s'].cuda()
                _val = batch_data['Y_d'].cuda()
                score = torch.add(_val, score)
                score, _index = torch.topk(score, self.beam)
                index = index[batch_data['idx'], _index]
            else:
                score, index = torch.topk(score, self.beam)
            batch_pred_time += time.time()-time_begin
            score = score.cpu().numpy()
            index = index.cpu().numpy()
            self._update_in_sparse(count, index, score, pred_lbs)
            if batch_idx % self.progress_step == 0:
                self.logger.info("Prediction progress: [{}/{}]".format(
                    batch_idx, num_batches))
            count += index.shape[0]
            del score, index
        pred_lbs = retain_topk(pred_lbs.tocsr(), copy=False, k=self.beam)
        self.net.cpu()
        avg_time = batch_pred_time*1000/pred_lbs.shape[0]
        self.logger.info("Avg pred time {:0.4f} ms".format(avg_time))
        if self.net.offset:
            pred_lbs = pred_lbs.tocsc()[:, :-1].tocsr()
        return pred_lbs.tocsr(), batch_pred_time

    def _train_depth(self, train_ds, valid_ds, model_dir,
                     depth, validate_after=5):
        """
        training function for module 1,2, and 4
        Arguments
            ----------
            train_ds: Class object
                training dataset
            valid_ds: Class object
                validation dataset
            model_dir: string
                Model directory for the outputs
            depth: int
                0: for module 1 and 2
                1: for module 4
            validate_after: int
                Validate after this many epochs starting from zero
        """
        train_dl, valid_dl = self._prep_for_depth(depth, train_ds, valid_ds)
        num_lbs = train_ds.labels.Y.shape[1]
        fname = os.path.join(model_dir, self.model_fname +
                             "-depth=%d_params.pkl" % (depth))
        if os.path.exists(fname):
            self.logger.info("Using pre-trained at "+str(depth))
            self.load(model_dir, depth)
            self.transfer_to_devices()
            return

        num_epochs = self.call_back[depth]
        counter_next_decay = self.dlr_step
        for epoch in range(0, num_epochs):
            if counter_next_decay == 0:
                self._adjust_parameters()
                counter_next_decay = self.dlr_step
            batch_train_start_time = time.time()
            tr_avg_loss = 0
            tr_avg_loss = self._step(train_dl)
            self.tracking.mean_train_loss.append(tr_avg_loss)
            batch_train_end_time = time.time()
            self.tracking.train_time = self.tracking.train_time + \
                batch_train_end_time - batch_train_start_time

            self.logger.info("Epoch: {}, loss: {}, time: {} sec".format(
                epoch, tr_avg_loss,
                batch_train_end_time - batch_train_start_time))

            if valid_dl is not None and epoch % validate_after == 0:
                if num_lbs < 600*1e3 or depth == 0:
                    self.validate(valid_dl, model_dir, epoch)
                self.save(model_dir, depth)
            counter_next_decay -= 1
            self.tracking.last_epoch += 1

        if num_lbs < 600*1e3 or depth == 0:
            self.validate(valid_dl, model_dir, epoch)
        self.save(model_dir, depth)
        return

    def _fit(self, train_ds, valid_ds, model_dir, result_dir, validate_after=5):
        """
            Fits model over the training data and trains module 1,2 and 4
            Arguments
                ----------
                train_ds: Class object
                    Training dataset
                valid_ds: Class object
                    Validation dataset
                model_dir: string
                    model directory
                result_dir: string
                    result directory
                validate_after: int
                    compute performane of validation dataset after
                    these many epochs
        """
        for depth in np.arange(len(self.batch_size)):
            self._train_depth(train_ds, valid_ds, model_dir,
                              depth, validate_after)
            torch.set_grad_enabled(False)
            torch.cuda.empty_cache()

        self.tracking.save(os.path.join(result_dir, 'training_statistics.pkl'))
        self.logger.info(
            "Training time: {} sec, Validation time: {} sec"
            ", Shortlist time: {} sec, Model size: {} MB".format(
                self.tracking.train_time, self.tracking.validation_time,
                self.tracking.shortlist_time, self.net.model_size))

    def create_label_fts_vec(self, train_dataset):
        """
            Computer BoW label centroids
            Arguments:
                ----------
                train_dataset: Class object
                    Training dataset
            Returns:
                CSR Matrix: L X V BoW sparse label centroids
        """
        _labels = train_dataset.labels.Y.transpose()
        _features = train_dataset.features.X
        return _labels.dot(_features)

    def fit(self, data_dir, model_dir, result_dir, dataset,
            data=None, tr_feat_fname='trn_X_Xf.txt',
            tr_label_fname='trn_X_Y.txt', val_feat_fname='tst_X_Xf.txt',
            val_label_fname='tst_X_Y.txt', num_workers=4, keep_invalid=False,
            feature_indices=None, label_indices=None, normalize_features=True,
            normalize_labels=False, validate=False, validate_after=5, **kwargs):
        """
        Fits network based on the inputs provided
        """
        self.logger.info("Loading training data.")
        train_dataset = self._create_dataset(
            os.path.join(data_dir, dataset),
            fname_features=tr_feat_fname,
            fname_labels=tr_label_fname,
            data=data, mode='train', keep_invalid=keep_invalid,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            feature_indices=feature_indices,
            label_indices=label_indices)

        if validate:
            self.logger.info("Loading validation data.")
            valid = self._create_dataset(
                os.path.join(data_dir, dataset),
                fname_features=val_feat_fname,
                fname_labels=val_label_fname,
                data={'X': None, 'Y': None}, mode='test',
                keep_invalid=keep_invalid,
                normalize_features=normalize_features,
                normalize_labels=normalize_labels,
                feature_indices=feature_indices,
                label_indices=label_indices)
            valid.labels.Y = self._filter_labels(valid.labels.Y)

        labels_repr = self.create_label_fts_vec(train_dataset)
        self.logger.info(f"Label centroids shape: {labels_repr.shape}")
        build_start = time.time()
        self.net.build(labels_repr, model_dir=model_dir)
        build_end = time.time()
        build_time = build_end-build_start
        self.tracking.train_time += build_time
        self._fit(train_dataset, valid, model_dir, result_dir, validate_after)

    def _format_acc(self, acc):
        _res = ""
        if isinstance(acc, dict):
            for key, val in acc.items():
                _res += "{}: {} ".format(key, val[0]*100)
        else:
            _res = "clf: {}".format(acc[0][0:5:2]*100)
        return _res

    def predict(self, data_dir, model_dir, dataset, data=None,
                ts_feat_fname='tst_X_Xf.txt', ts_label_fname='tst_X_Y.txt',
                batch_size=256, num_workers=6, keep_invalid=False,
                feature_indices=None, label_indices=None,
                normalize_features=True, normalize_labels=False, **kwargs):
        """
            Wrapper to make prediction of the input dataset
            Returns
                ----------
                CSR Matrix: returns NxL sparse matrix where each row
                            contains top-k elements
        """
        self.net.build(None, model_dir)
        dataset = self._create_dataset(
            os.path.join(data_dir, dataset), fname_features=ts_feat_fname,
            fname_labels=ts_label_fname, data=data,
            keep_invalid=keep_invalid, normalize_features=normalize_features,
            normalize_labels=normalize_labels, mode='test',
            feature_indices=feature_indices, label_indices=label_indices)
        predicted_labels, _ = self._predict(dataset, model_dir, **kwargs)
        self._print_stats(dataset.labels.Y, predicted_labels)
        return predicted_labels

    def _predict(self, dataset, model_dir, **kwargs):
        """
            Predicts most relevant items for each document in the dataset
            Arguments
                ----------
                dataset: Class Object
                model_dir: string
                    Model directory
            Returns:
                ----------
                CSR Matrix
                    Relevant items for each document
        """
        self.logger.info("Preping the classifile")
        for depth in range(0, self.net.height):
            self.logger.info("Predicting for depth {}".format(depth))
            self._prep_ds(depth, dataset)
            self.net._prep_for_depth(depth, initialize=False)
            self.load(model_dir, depth)
        data_loader = self._prep_dl(depth, dataset)
        return self._predict_depth(depth, data_loader)

    def extract(self, model_dir):
        """
            Get word embeddings from module 1
            Arguments
                ----------
                model_dir: string
                    model directory
            Returns
                ----------
                dict
                    word embeddings for the module 1
        """
        self.net.load(fname=model_dir)
        self.net._prep_for_depth(0, initialize=False)
        self.load(model_dir, 0)
        torch.set_grad_enabled(False)
        torch.cuda.empty_cache()
        embeddings = self.net.depth_node.embed.cpu().state_dict()
        return embeddings
    
    def get_document(self, data_dir, model_dir, dataset, data=None,
                     ts_feat_fname='tst_X_Xf.txt', batch_size=256,
                     num_workers=6, keep_invalid=False, if_labels=False,
                     feature_indices=None, label_indices=None,
                     normalize_features=True, normalize_labels=False, **kwargs):
        self.net.load(fname=model_dir)
        self.net._prep_for_depth(0, initialize=False)
        self.load(model_dir, 0)
        dataset = self._create_dataset(
            os.path.join(data_dir, dataset), fname_features=ts_feat_fname,
            fname_labels=ts_feat_fname, data=data,
            keep_invalid=keep_invalid, normalize_features=normalize_features,
            normalize_labels=normalize_labels, mode='test',
            feature_indices=feature_indices, label_indices=label_indices)
        self.net.cpu()
        print(self.net)
        self.net.to()
        docs = self._doc_embed(dataset, 0, self.net.depth_node, if_labels)
        return docs

    def _doc_embed(self, dataset, depth, encoder, return_coarse=False, **kwargs):
        encoder.to()
        torch.set_grad_enabled(False)
        torch.cuda.empty_cache()
        encoder.eval()
        data_loader = self._prep_dl(depth, dataset, mode='test')
        num_batches = len(data_loader)
        docs = []
        for batch_idx, batch_data in enumerate(data_loader):
            docs.append(encoder.encode(
                batch_data, return_coarse).cpu().numpy())
            if batch_idx % self.progress_step == 0:
                self.logger.info("Fectching progress: [{}/{}]".format(
                    batch_idx, num_batches))
        encoder.cpu()
        docs = np.vstack(docs)
        return docs

    def _doc_embed(self, dataset, depth=0, return_coarse=False, encoder=None):
        """
            Encodes document embeddings using the encoder block
            Arguments
                ----------
                dataset: Class Object
                    dataset for which document embeddings is required
                depth: int
                    0: module 1,2
                    1: module 4
                return_coarse: bool
                    If True residual block is not applied
                encoder: nn.Module
                    Document encoder block
        """
        encoder.cpu()
        encoder.eval()
        torch.set_grad_enabled(False)
        torch.cuda.empty_cache()
        encoder.to()
        data_loader = self._prep_dl(depth, dataset)
        num_batches = len(data_loader)
        docs = []
        for batch_idx, batch_data in enumerate(data_loader):
            docs.append(encoder.encode(
                batch_data, return_coarse=return_coarse).cpu().numpy())
            if batch_idx % self.progress_step == 0:
                self.logger.info("Fectching progress: [{}/{}]".format(
                    batch_idx, num_batches))
        encoder.cpu()
        docs = np.vstack(docs)
        return docs

    def _update_in_sparse(self, counter, predicted_label_index,
                          predicted_label_scores, sparse_mat):
        """
            Updates entries in sparse matrix
        """
        index = np.arange(
            predicted_label_index.shape[0]).reshape(-1, 1) + counter
        sparse_mat[index, predicted_label_index] = predicted_label_scores

    def _adjust_parameters(self):
        """
            Decaf rate scheudling
        """
        self.optimizer.adjust_lr(self.dlr_factor)
        self.learning_rate *= self.dlr_factor
        self.dlr_step = max(5, self.dlr_step//2)
        self.logger.info(
            "Adjusted learning rate to: {}".format(self.learning_rate))

    def save(self, model_dir, depth, *args):
        """
            Saving model parameters
        """
        fname = self.model_fname + "-depth=%d" % (depth)
        file_path = os.path.join(model_dir, fname+"_params.pkl")
        torch.save(self.net.state_dict(), file_path)

    def load(self, model_dir, depth,
             label_padding_index=None, *args):
        """
            Loading model parameters
        """
        fname = self.model_fname + "-depth=%d" % (depth)
        file_path = os.path.join(model_dir, fname+"_params.pkl")
        self.net.load_state_dict(torch.load(file_path))
        self.transfer_to_devices()

    def _evaluate(self, true_labels, predicted_labels):
        """
            Evaluate predicted matrix
        """
        pmat = predicted_labels.tocsr()
        acc = xc.Metrics(true_labels)
        rec = xc.recall(pmat, true_labels, self.beam)
        _p, _n = acc.eval(predicted_labels.tocsr(), self.beam)
        return _p, _n, rec

    def evaluate(self, true_labels, predicted_labels, _filter=True):
        """
            Wrapper function for evaluating prediction matrix
            Arguments
                ----------
                true_labels: CSR Matrix
                    ground truth for the prediction matrix
                predicted_labels: CSR matrix or dict of CSR Matrix
                    if dict each matrix in the dict is evaluated
                        against the ground truth
                    else evaluation is done only for the provided CSR matrix
                _filter: Bool
                    if True filter file is applied for the prediction matrix
            Returns
                ----------
                tuple or dict of tuple
                    Depends on the input type of predict_labels
                    each tuple consits of precision, ndcg and recall values
        """
        if self.net.depth == self.net.height - 1 and _filter:
            true_labels = self._filter_labels(true_labels)
            _predicted_labels = self._filter_labels(predicted_labels)
        else:
            _predicted_labels = predicted_labels
        if issparse(_predicted_labels):
            return self._evaluate(true_labels,
                                  _predicted_labels)
        else:  # Multiple set of predictions
            acc = {}
            for key, val in _predicted_labels.items():
                acc[key] = self._evaluate(true_labels, val)
            return acc

    def _filter_labels(self, score_mat):
        """
            Filter datapoints in the score mat
        """
        if len(self.filter_docs) > 0:
            self.logger.info("Filtering labels to remove overlap")
            if issparse(score_mat):
                return_score_mat = score_mat.copy().tolil()
                return_score_mat[self.filter_docs, self.filter_lbls] = 0
                return_score_mat = return_score_mat.tocsr()
            else:
                return_score_mat = {}
                for key, _ in score_mat.items():
                    return_score_mat[key] = score_mat[key].copy().tolil()
                    return_score_mat[key][self.filter_docs,
                                          self.filter_lbls] = 0
                    return_score_mat[key] = return_score_mat[key].tocsr()
            return return_score_mat
        else:
            return score_mat
