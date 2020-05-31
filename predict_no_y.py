# Usage: python predict_no_y.py  number_of_separation NEPDF_pathway model_pathway
# command line in developer's linux machine :
# python predict_no_y.py  9 /home/yey3/cnn_project/code3/NEPDF_data   /home/yey3/cnn_project/code3/trained_model/models/KEGG_keras_cnn_trained_model_shallow2.h5
from __future__ import print_function
import json

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
import os,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
################################
# Jean modifications
import argparse

from networks import make_dnn, make_cnnc
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
        "--fit-dnn", action="store_true", default=False, help="fit DNN vs CNNC"
    )
    parser.add_argument(
        "--n-layers", type=int, default=2, help="Number of hidden layers"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=10, help="Number of hidden nodes per layer"
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.15, help="probability of dropping out a node"
    )
    parser.add_argument(
        "--do-binary", action="store_true", default=False, help="fit binary outcome"
    )
    parser.add_argument(
        "--data-path", type=str
    )
    parser.add_argument(
        "--num-classes", type=int, default=3
    )
    parser.add_argument(
        "--model-path", type=str
    )
    parser.add_argument(
        "--tf-idx", type=int, default=1
    )
    parser.add_argument(
        "--out-file", type=str
    )
    args = parser.parse_args()

    if args.num_classes == 2:
        assert args.do_binary

    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)
    test_TF = [args.tf_idx]
    x_test, y_test, _ = load_data_TF2(test_TF,args.data_path, binary_outcome=args.do_binary, flatten=args.fit_dnn)
    print(x_test.shape, 'x_test samples')
    ############

    if args.fit_dnn:
        model = make_dnn(x_test, args.num_classes, num_layers=args.n_layers, num_hidden=args.n_hidden, dropout_rate=args.dropout_rate)
    else:
        model = make_cnnc(x_test, args.num_classes)
    model.load_weights(args.model_path)
    ###########################################################
    print ('predict')
    y_predict = model.predict(x_test)
    perf_dict = get_perf(y_predict, y_test)
    perf_dict['model'] = 'DNN' if args.fit_dnn else 'CNNC'
    print(perf_dict)
    with open(args.out_file, 'w') as f:
        json.dump(perf_dict, f)

if __name__ == "__main__":
    main(sys.argv[1:])
