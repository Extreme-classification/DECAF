import argparse
import json
import libs.utils as utils
import numpy as np

__author__ = 'X'


class Parameters():
    """
        Class for parameters in XML
    """

    def __init__(self, description):
        self.parser = argparse.ArgumentParser(description)
        self.params = None
        self._construct()

    def _construct(self):
        self.parser.add_argument(
            '--data_dir',
            dest='data_dir',
            action='store',
            type=str,
            help='path to main data directory')
        self.parser.add_argument(
            '--dataset',
            dest='dataset',
            action='store',
            type=str,
            help='dataset name')
        self.parser.add_argument(
            '--model_dir',
            dest='model_dir',
            action='store',
            type=str,
            help='directory to store models')
        self.parser.add_argument(
            '--result_dir',
            dest='result_dir',
            action='store',
            type=str,
            help='directory to store results')
        self.parser.add_argument(
            '--emb_dir',
            action='store',
            type=str,
            default="random",
            help='directory of word embeddings')
        self.parser.add_argument(
            '--model_fname',
            dest='model_fname',
            default='model',
            action='store',
            type=str,
            help='model file name')
        self.parser.add_argument(
            '--config',
            dest='config',
            default='base.json',
            action='store',
            type=str,
            help='model config files')
        self.parser.add_argument(
            '--mode',
            dest='mode',
            default='predict',
            action='store',
            type=str,
            help='model mode')
        self.parser.add_argument(
            '--tree_idx',
            dest='tree_idx',
            default=-1,
            action='store',
            type=int,
            help='model instance id')
        self.parser.add_argument(
            '--pred_fname',
            dest='pred_fname',
            default="test_predictions",
            action='store',
            type=str,
            help='prediction fname')
            

    def parse_args(self):
        self.params = self.parser.parse_args()
        _json = utils.fetch_json(self.params.config, self.params)
        for key, val in _json["DEFAULT"].items():
            self.params.__dict__[key] = val
        for key, val in _json[self.params.model_fname].items():
            self.params.__dict__[key] = val

        for key in _json["SETUP"].keys():
            over = _json["SETUP"][key]["over"]
            action = _json["SETUP"][key]["action"]
            if action == "size":
                _data = np.loadtxt(self.params.__dict__[over])
                self.params.__dict__[key] = _data.size

    def load(self, fname):
        vars(self.params).update(json.load(open(fname)))

    def save(self, fname):
        print(vars(self.params))
        json.dump(vars(self.params), open(fname, 'w'), indent=4)
