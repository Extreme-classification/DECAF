import numpy as np
import scipy.sparse as sp
import json
import os
import re


def load_overlap(data_dir, valid_labels, filter_label_file='filter_labels'):
    docs = np.asarray([])
    lbs = np.asarray([])
    if os.path.exists(os.path.join(data_dir, filter_label_file)):
        print("Loading from pre-build file")
        filter_lbs = np.loadtxt(os.path.join(
            data_dir, filter_label_file), dtype=np.int32)
        if filter_lbs.size > 0:
            docs = filter_lbs[:, 0]
            lbs = filter_lbs[:, 1]
            valid_labels = np.loadtxt(valid_labels, dtype=np.int32)
            max_lbl = max(max(lbs), max(valid_labels))+1
            hash_key = np.ones(max_lbl)*-1
            hash_key[valid_labels] = np.arange(valid_labels.size)
            lbs_remapped = hash_key[lbs]
            valid_idx = np.where(lbs_remapped != -1)[0]
            docs = docs[valid_idx]
            lbs = lbs_remapped[valid_idx]
    print("Overlap is:", docs.size)
    return docs, lbs


def append_padding_embedding(embeddings):
    """
        Append a row of zeros as embedding for <PAD>
        Args:
            embeddings: numpy.ndarray: embedding matrix
        Returns:
            embeddings: numpy.ndarray: transformed embedding matrix
    """
    embedding_dim = embeddings.shape[1]
    app = np.zeros((1, embedding_dim))
    return np.vstack([embeddings, app])


def save_predictions(preds, result_dir, prefix='predictions'):
    sp.save_npz(os.path.join(result_dir, '{}.npz'.format(prefix)), preds)


def resolve_schema_args(jfile, ARGS):
    """
        Reads JSON and complete the parameters from ARGS
    """
    arguments = re.findall(r"#ARGS\.(.+?);", jfile)
    for arg in arguments:
        replace = '#ARGS.%s;' % (arg)
        to = str(ARGS.__dict__[arg])
        if jfile.find('\"#ARGS.%s;\"' % (arg)) != -1:
            replace = '\"#ARGS.%s;\"' % (arg)
            if isinstance(ARGS.__dict__[arg], str):
                to = str("\""+ARGS.__dict__[arg]+"\"")
        jfile = jfile.replace(replace, to)
    return jfile

def fetch_json(file, ARGS):
    with open(file, encoding='utf-8') as f:
        file = ''.join(f.readlines())
        schema = resolve_schema_args(file, ARGS)
    return json.loads(schema)
