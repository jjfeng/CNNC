import argparse
import json
import sys
import logging
import joblib
import glob

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns

from predict_easiernet import load_easier_net


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
        "--model-path", type=str
    )
    parser.add_argument(
        "--tf-idx", type=int, default=1
    )
    parser.add_argument(
        "--is-vgg", action="store_true",
    )
    parser.add_argument(
        "--plot-support-file", type=str
    )
    parser.add_argument(
        "--log-file", type=str
    )
    args = parser.parse_args()

    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )

    models = load_easier_net(args.model_path, args.is_vgg)
    all_supports = []
    for model in models:
        all_supports.append(model.support())
    all_supports = np.array(all_supports)
    input_size = int(np.sqrt(all_supports.shape[1]))
    support_mean = all_supports.mean(axis=0).reshape((input_size, input_size))

    sns.heatmap(support_mean)
    plt.title("TF %d, EASIER-net support" % args.tf_idx)
    plt.savefig(args.plot_support_file)






if __name__ == "__main__":
    main(sys.argv[1:])
