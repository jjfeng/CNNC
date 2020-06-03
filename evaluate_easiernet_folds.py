import argparse
import sys
import json
import pickle
import logging

import numpy as np
import pandas as pd

import torch

from spinn2.evaluate_siernet_folds import eval_fold_models
from spinn2.network import SierNet, VGGSierNet

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
    parser.add_argument("--model-path", type=str, default="_output/nn.pt")
    parser.add_argument(
        "--is-vgg", action="store_true",
    )
    parser.add_argument("--fold-idxs-file", type=str, default=None)
    parser.add_argument(
        "--data-path", type=str
    )
    parser.add_argument(
        "--num-tf", type=int, default=2
    )
    parser.add_argument(
        "--exclude-tf", type=int, default=1
    )
    parser.add_argument(
        "--do-binary", action="store_true", default=False, help="fit binary outcome"
    )
    parser.add_argument("--out-file", type=str, default="_output/eval.json")
    parser.add_argument("--log-file", type=str, default="_output/eval_nn.txt")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def load_easier_nets(model_file, is_vgg):
    meta_state_dict = torch.load(model_file)
    all_models = []
    for fold_dicts in meta_state_dict["state_dicts"]:
        init_models = []
        for fold_state_dict in fold_dicts:
            if is_vgg:
                model = VGGSierNet(
                    n_inputs=(meta_state_dict["n_inputs"],meta_state_dict["n_inputs"]),
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
            model.load_state_dict(fold_state_dict)
            init_models.append(model)
        model.get_net_struct()
        all_models.append(init_models)
    return all_models, meta_state_dict


def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    logging.info(str(args))
    np.random.seed(args.seed)

    #####
    # Load data
    #####
    x_trains = []
    y_trains = []
    whole_data_TF = [i for i in range(args.num_tf) if i != args.exclude_tf]
    for tf_idx in whole_data_TF:
        x_train, y_train, _ = load_data_TF2([tf_idx],args.data_path,binary_outcome=args.do_binary,flatten=not args.is_vgg)
        x_trains.append(x_train)
        y_trains.append(y_train)
    if args.is_vgg:
        x_trains = [x.reshape((x.shape[0], 1, x.shape[1], x.shape[2])) for x in x_trains]

    # Load folds
    with open(args.fold_idxs_file, "rb") as f:
        fold_idx_dicts = pickle.load(f)
        num_folds = len(fold_idx_dicts)

    # Load models and evaluate them on folds, take the average
    all_models, meta_state_dict = load_easier_nets(args.model_path, args.is_vgg)
    all_losses = []
    for fold_idx, fold_dict in enumerate(fold_idx_dicts):
        test_x = np.concatenate([x_trains[i] for i in fold_dict["test"]], axis=0)
        test_y = np.concatenate([y_trains[i] for i in fold_dict["test"]], axis=0).reshape((-1,1))
        fold_models = [seed_fold_models[fold_idx] for seed_fold_models in all_models]
        empirical_loss = eval_fold_models(test_x, test_y, fold_models)
        all_losses.append(empirical_loss)
    avg_loss = np.mean(all_losses)

    # Store the ensemble results
    meta_state_dict.pop("state_dicts", None)
    meta_state_dict.pop("weight", None)
    meta_state_dict["cv_loss"] = float(avg_loss)
    meta_state_dict["seed_losses"] = list(map(float, all_losses))
    print(meta_state_dict)
    json.dump(meta_state_dict, open(args.out_file, "w"))


if __name__ == "__main__":
    main(sys.argv[1:])

