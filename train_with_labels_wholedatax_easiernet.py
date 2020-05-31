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
import argparse
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
    parser.add_argument("--data-file", type=str, default="_output/data.npz")
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
        "--fit-dnn", action="store_true", default=False, help="Fit DNN vs CNNC"
    )
    parser.add_argument(
        "--data-path", type=str
    )
    parser.add_argument(
        "--num-tf", type=int, default=1
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
    parser.add_argument("--log-file", type=str, default="_output/log_nn.txt")
    parser.add_argument("--out-model-file", type=str, default="_output/nn.pt")
    args = parser.parse_args()

    assert args.num_classes != 1

    return args

def _fit(
    estimator,
    X,
    y,
    train,
    max_iters: int = 100,
    max_prox_iters: int = 100,
    seed: int = 0,
) -> list:
    torch.manual_seed(seed)
    X_train = X[train]
    y_train = y[train]

    my_estimator = clone(estimator)
    my_estimator.fit(
        X_train, y_train, max_iters=max_iters, max_prox_iters=max_prox_iters
    )
    return my_estimator

def load_data_TF2(indel_list,data_path, flatten=False): # cell type specific  ## random samples for reactome is not enough, need borrow some from keggp
    import random
    import numpy as np
    xxdata_list = []
    yydata = []
    count_set = [0]
    count_setx = 0
    for i in indel_list:#len(h_tf_sc)):
        xdata = np.load(data_path+'/Nxdata_tf' + str(i) + '.npy')
        ydata = np.load(data_path+'/ydata_tf' + str(i) + '.npy')
        for k in range(len(ydata)):
            if flatten:
                xxdata_list.append(xdata[k,:,:,:].flatten())
            else:
                xxdata_list.append(xdata[k,:,:,:])
            yydata.append(ydata[k])
        count_setx = count_setx + len(ydata)
        count_set.append(count_setx)
        print (i,len(ydata))
    yydata_array = np.array(yydata)
    yydata_x = yydata_array.astype('int')
    print (np.array(xxdata_list).shape)
    return((np.array(xxdata_list),yydata_x,count_set))

def main(args=sys.argv[1:]):
    args = parse_args(args)
    if len(sys.argv) < 4:
        print ('No enough input files')
        sys.exit()

    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )

    whole_data_TF = [i for i in range(args.num_tf)]
    (x_train, y_train,count_set_train) = load_data_TF2(whole_data_TF,args.data_path, flatten=True)
    n_obs = x_train.shape[0]
    n_inputs = x_train.shape[1]
    print(x_train.shape, 'x_train samples')
    save_dir = os.path.join(os.getcwd(), '_output', 'whole_model_test')
    print(y_train)
    #if num_classes > 2:
    #    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_train = y_train.reshape((y_train.size, 1))
    print(y_train.shape, 'y_train samples')
        ###########
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        ############
    #torch.manual_seed(seed)
    #print("THREADS", torch.get_num_threads())
    base_estimator = SierNetEstimator(
        n_inputs=n_inputs,
        input_filter_layer=True,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_out=args.num_classes,
        full_tree_pen=args.full_tree_pen,
        input_pen=args.input_pen,
        batch_size=args.batch_size, #(n_obs//n_batches + 1),
        num_classes=args.num_classes,
        # Weight classes by inverse of their observed ratios. Trying to balance classes
        weight=(n_obs / (args.num_classes * np.bincount(y_train.flatten()))
        if args.num_classes >= 2
        else None),
    )
    n_jobs = args.num_inits
    parallel = Parallel(n_jobs=n_jobs, verbose=True, pre_dispatch=n_jobs)
    all_estimators = parallel(delayed(_fit)(
            base_estimator,
            x_train,
            y_train,
            train=np.arange(x_train.shape[0]),
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
