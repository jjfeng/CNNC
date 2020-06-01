# Usage: python predict_no_y.py  number_of_separation NEPDF_pathway model_pathway
# command line in developer's linux machine :
# python predict_no_y.py  9 /home/yey3/cnn_project/code3/NEPDF_data   /home/yey3/cnn_project/code3/trained_model/models/KEGG_keras_cnn_trained_model_shallow2.h5
from __future__ import print_function
import os,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
###############################
# Jean modifications
import argparse
import json
import logging

import torch
from scipy.special import logsumexp

from spinn2.network import SierNet, VGGSierNet

from common import get_perf
from train_with_labels_wholedatax_easiernet import load_data_TF2

def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=12,
    )
    parser.add_argument(
        "--do-binary", action="store_true", default=False, help="fit binary outcome"
    )
    parser.add_argument(
        "--data-path", type=str
    )
    parser.add_argument(
        "--model-path", type=str
    )
    parser.add_argument(
        "--tf-idx", type=int, default=1
    )
    parser.add_argument(
        "--is-vgg", action="store_true",
    )
    parser.add_argument(
        "--out-file", type=str
    )
    parser.add_argument(
        "--log-file", type=str
    )
    args = parser.parse_args()

    return args

def load_easier_net(model_file, is_vgg):
    meta_state_dict = torch.load(model_file)
    models = []
    for state_dict in meta_state_dict["state_dicts"]:
        if is_vgg:
            model = VGGSierNet(
                n_inputs=meta_state_dict["n_inputs"],
                n_out=meta_state_dict["n_out"],
                input_filter_layer=meta_state_dict["input_filter_layer"],
            )
        else:
            model = SierNet(
                n_layers=meta_state_dict["n_layers"],
                n_input=meta_state_dict["n_inputs"],
                n_hidden=meta_state_dict["n_hidden"],
                n_out=meta_state_dict["n_out"],
                input_filter_layer=meta_state_dict["input_filter_layer"],
            )
        model.load_state_dict(state_dict)
        model.eval()
        model.get_net_struct()
        models.append(model)
    return models

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )

    test_TF = [args.tf_idx]
    (x_test, y_test, count_set) = load_data_TF2(test_TF,args.data_path, binary_outcome=args.do_binary, flatten=not args.is_vgg)
    if args.is_vgg:
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]))
    else:
        x_test = x_test.reshape((x_test.shape[0],-1))
    print(x_test.shape, 'x_test samples')
    y_test = y_test.reshape((y_test.size, 1))
    ############

    models = load_easier_net(args.model_path, args.is_vgg)

    y_log_prob = logsumexp([model.predict_log_proba(x_test) for model in models], axis=0) - np.log(len(models))
    y_predict = np.exp(y_log_prob)
    perf_dict = get_perf(y_predict, y_test)
    perf_dict['model'] = "EASIERnet-DNN" if not args.is_vgg else "EASIERnet-VGG"
    print(perf_dict)
    with open(args.out_file, 'w') as f:
        json.dump(perf_dict, f)

if __name__ == "__main__":
    main(sys.argv[1:])
