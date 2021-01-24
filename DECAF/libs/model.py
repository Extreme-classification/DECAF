from .model_base import ModelDECAFBase
import libs.features as feat


class ModelDECAF(ModelDECAFBase):
    """
        class ModelDECAF
        Implements base class for module 1.
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
        super(ModelDECAF, self).__init__(
            params, net, criterion, optimizer, *args, **kwargs)


class ModelDECAFpp(ModelDECAFBase):
    """
        class ModelDECAFpp
        Implements base class for module 2,3,4 for DECAF and DECAF-v.
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
        super(ModelDECAFpp, self).__init__(
            params, net, criterion, optimizer, *args, **kwargs)

    def create_label_fts_vec(self, train_dataset):
        """
            Creates dense label centroids embeddings
            Arguemnts:
                train_dataset: Class object
                    training dataset
            Returns:
                np.ndarray
                    Num labels X embeddings dims label centroids
        """
        model = self.net.get_only_encoder()
        train_dataset.mode = "test"
        doc_repr = self._doc_embed(train_dataset, 0, True, model)
        train_dataset.mode = "train"
        _labels = train_dataset.labels.Y.transpose()
        labels_repr = _labels.dot(doc_repr)
        return labels_repr


class ModelDECAFfe(ModelDECAFpp):
    """
        class ModelDECAFpp
        Implements base class for module 2,3,4 for DECAF-lite.
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
        super(ModelDECAFfe, self).__init__(
            params, net, criterion, optimizer, *args, **kwargs)

    def _prep_for_depth(self, depth, train_ds, valid_ds):
        """
            Creates dataloader for train module 2 as well as module 4
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
                nn.Dataloader, nn.Dataloader
                    for training and validation dataloader respectively
        """
        train_ds.mode = 'test'
        self._prep_ds(depth, train_ds)
        self._prep_ds(depth, valid_ds)
        self.net.cpu()
        self.net._prep_for_depth(depth)
        print(self.net)
        if depth == 0:
            document = self._doc_embed(train_ds, 0, True, self.net.depth_node)
            train_ds.feature_type = "dense"
            train_ds.features = feat.construct(
                "", "", document, False, "dense")

            document = self._doc_embed(valid_ds, 0, True, self.net.depth_node)
            valid_ds.feature_type = "dense"
            valid_ds.features = feat.construct(
                "", "", document, False, "dense")

        self.optimizer.learning_rate = self.lrs[depth]
        self.learning_rate = self.lrs[depth]
        self.optimizer.construct(self.net.depth_node, None)
        self.dlr_step = self.dlr_steps[depth]
        train_ds.mode = 'train'
        train_dl = self._prep_dl(depth, train_ds, mode='train')
        valid_dl = self._prep_dl(depth, valid_ds, mode='test')
        return train_dl, valid_dl

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
            if depth == 0:
                document = self._doc_embed(dataset, depth, True,
                                           encoder=self.net.depth_node)
                dataset.feature_type = "dense"
                dataset.features = feat.construct(
                    "", "", document, False, "dense")

        data_loader = self._prep_dl(depth, dataset)
        return self._predict_depth(depth, data_loader)
