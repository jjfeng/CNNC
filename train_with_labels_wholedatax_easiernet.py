from __future__ import print_function
# Usage  python train_with_labels_wholedata.py number_of_data_parts_divided
# command line in developer's linux machine :
# module load cuda-8.0 using GPU
#srun -p gpu --gres=gpu:1 -c 2 --mem=20Gb python train_with_labels_wholedatax.py 9 /home/yey3/cnn_project/code3/NEPDF_data 3 > results_whole.txt
#######################OUTPUT
# it will generate a folder 'wholeXXXXX', in which 'keras_cnn_trained_model_shallow.h5' is the final trained model
#import keras
#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#from keras.optimizers import SGD
#from keras.callbacks import EarlyStopping,ModelCheckpoint
import os,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp

###############
# Jean modifications
import json
import pickle
import argparse
import itertools
from joblib import Parallel, delayed
import logging
from sklearn.base import clone
import torch

from spinn2.sier_net import SierNetEstimator

###################################################
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
        "--num-inits",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=0,
        help="Number of classes in classification. Should be zero if doing regression",
    )
    parser.add_argument(
        "--do-binary", action="store_true", default=False, help="fit binary outcome"
    )
    parser.add_argument(
        "--data-path", type=str
    )
    parser.add_argument(
        "--fold-idxs-file", type=str
    )
    parser.add_argument(
        "--num-tf", type=int, default=2
    )
    parser.add_argument(
        "--exclude-tf", type=int, default=1
    )
    parser.add_argument(
        "--batch-size", type=int, default=32
    )
    parser.add_argument(
        "--n-layers", type=int, default=3, help="Number of hidden layers"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=50, help="Number of hidden nodes per layer"
    )
    parser.add_argument(
        "--full-tree-pen", type=float, default=0.001
    )
    parser.add_argument(
        "--input-pen", type=float, default=0.001
    )
    parser.add_argument(
        "--max-prox-iters", type=int, default=40, help="Number of prox epochs"
    )
    parser.add_argument(
        "--max-iters", type=int, default=40, help="Number of Adam epochs"
    )
    parser.add_argument(
        "--model-fit-params-file",
        type=str,
        help="A json file that specifies what the hyperparameters are. If given, this will override the arguments passed in.",
    )
    parser.add_argument("--log-file", type=str, default="_output/log_nn.txt")
    parser.add_argument("--out-model-file", type=str, default="_output/nn.pt")
    args = parser.parse_args()

    assert args.num_classes != 1
    if args.model_fit_params_file is not None:
        with open(args.model_fit_params_file, "r") as f:
            model_params = json.load(f)
            args.full_tree_pen = model_params["full_tree_pen"]
            args.input_pen = model_params["input_pen"]
            args.input_filter_layer = model_params["input_filter_layer"]
            args.n_layers = model_params["n_layers"]
            args.n_hidden = model_params["n_hidden"]

    args.n_jobs = args.num_inits
    return args

def _fit(
    estimator,
    X_trains,
    y_trains,
    train,
    max_iters: int = 100,
    max_prox_iters: int = 100,
    seed: int = 0,
) -> list:
    torch.manual_seed(seed)
    X_train = np.concatenate([X_trains[i] for i in train], axis=0)
    y_train = np.concatenate([y_trains[i] for i in train], axis=0)
    y_train = y_train.reshape((y_train.size, 1))

    my_estimator = clone(estimator)
    my_estimator.fit(
        X_train, y_train, max_iters=max_iters, max_prox_iters=max_prox_iters
    )
    return my_estimator

def load_data_TF2(indel_list, data_path, binary_outcome=False, flatten=False): # cell type specific  ## random samples for reactome is not enough, need borrow some from keggp
    import random
    import numpy as np
    xxdata_list = []
    yydata = []
    count_set = [0]
    count_setx = 0
    for i in indel_list:
        xdata = np.load(data_path+'/Nxdata_tf' + str(i) + '.npy')
        ydata = np.load(data_path+'/ydata_tf' + str(i) + '.npy')
        num_obs = 0
        for k in range(len(ydata)):
            if binary_outcome and int(ydata[k]) > 1:
                continue
            if flatten:
                xxdata_list.append(xdata[k,:,:,:].flatten())
            else:
                xxdata_list.append(xdata[k,:,:,:])
            yydata.append(ydata[k])
            num_obs += 1
        count_setx = count_setx + num_obs
        count_set.append(count_setx)
        print (i,len(ydata))
    yydata_array = np.array(yydata)
    yydata_x = yydata_array.astype('int')
    return((np.array(xxdata_list),yydata_x,count_set))

def main(args=sys.argv[1:]):
    args = parse_args(args)
    if len(sys.argv) < 4:
        print ('No enough input files')
        sys.exit()

    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )

    #####
    # Load data
    #####
    x_trains = []
    y_trains = []
    whole_data_TF = [i for i in range(args.num_tf) if i != args.exclude_tf]
    for tf_idx in whole_data_TF:
        x_train, y_train, _ = load_data_TF2([tf_idx],args.data_path,binary_outcome=args.do_binary,flatten=True)
        x_trains.append(x_train)
        y_trains.append(y_train)
    n_inputs = x_train.shape[1]
    n_obs = sum([x_train.shape[0] for x_train in x_trains])
    y = np.concatenate(y_trains)

    ######
    # Begin training
    ######
    base_estimator = SierNetEstimator(
        n_inputs=n_inputs,
        input_filter_layer=True,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_out=args.num_classes,
        full_tree_pen=args.full_tree_pen,
        input_pen=args.input_pen,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        weight=n_obs / (args.num_classes * np.bincount(y.flatten()))
        if args.num_classes >= 2
        else None,
    )
    parallel = Parallel(n_jobs=args.n_jobs, verbose=True, pre_dispatch=args.n_jobs)
    if args.fold_idxs_file is not None:
        with open(args.fold_idxs_file, "rb") as f:
            fold_idx_dict = pickle.load(f)
            num_folds = len(fold_idx_dict)

        all_estimators = parallel(
            delayed(_fit)(
                base_estimator,
                x_trains,
                y_trains,
                train=fold_idx_dict[fold_idx]["train"],
                max_iters=args.max_iters,
                max_prox_iters=args.max_prox_iters,
                seed=args.seed + num_folds * init_idx + fold_idx,
            )
            for fold_idx, init_idx in itertools.product(
                range(num_folds), range(args.num_inits)
            )
        )

        # Just printing things from the first fold
        logging.info(f"sample estimator 0 fold 0")
        all_estimators[0].net.get_net_struct()

        assert (num_folds * args.num_inits) == len(all_estimators)

        meta_state_dict = all_estimators[0].get_params()
        meta_state_dict["state_dicts"] = [
            [None for _ in range(num_folds)] for _ in range(args.num_inits)
        ]
        for (fold_idx, init_idx), estimator in zip(
            itertools.product(range(num_folds), range(args.num_inits)), all_estimators
        ):
            meta_state_dict["state_dicts"][init_idx][
                fold_idx
            ] = estimator.net.state_dict()
        torch.save(meta_state_dict, args.out_model_file)
    else:
        all_estimators = parallel(delayed(_fit)(
                base_estimator,
                x_trains,
                y_trains,
                train=np.arange(len(x_trains)),
                max_iters=args.max_iters,
                max_prox_iters=args.max_prox_iters,
                seed=args.seed + init_idx,
            )
            for init_idx in range(args.num_inits)
        )
        meta_state_dict = all_estimators[0].get_params()
        meta_state_dict["state_dicts"] = [
            estimator.net.state_dict() for estimator in all_estimators
        ]
        print("SUCCESS")
        torch.save(meta_state_dict, args.out_model_file)

if __name__ == "__main__":
    main(sys.argv[1:])
