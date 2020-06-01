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
from matplotlib.lines import Line2D


METHOD_DICT_NAME = {
    "DNN": "DNN",
}


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "table_file", type=str,
    )
    parser.add_argument(
        "out_plot", type=str,
    )
    parser.add_argument(
        "--yval", type=str, default="test_loss"
    )
    parser.add_argument(
        "--ylab", type=str, default=None
    )
    parser.add_argument(
        "--ymin", type=float, default=0.5,
    )
    parser.add_argument(
        "--ymax", type=float, default=0.85,
    )
    parser.set_defaults()
    args = parser.parse_args()

    if args.ylab is None:
        args.ylab = args.yval

    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)

    # Load model results
    res_df = pd.read_csv(args.table_file, index_col=0)
    res_df = res_df[["tf", "model", args.yval]]
    res_df.columns = ["TF", "Model", args.ylab]
    res_df = res_df.astype({'TF': 'int32'})

    palette = sns.color_palette()

    order = (
        res_df[["TF", args.ylab]].groupby("TF")
        .min()
        .reset_index()
        .sort_values(by=args.ylab)["TF"]
    )
    print(order)
    order_dict = {order.iloc[i]: i for i in range(order.size)}
    res_df["Transcription factor"] = [
        order_dict[a] for a in res_df["TF"]
    ]

    sns.set_context("paper", font_scale=1.2)
    grid = sns.stripplot(
        x="Transcription factor",
        y=args.ylab,
        hue="Model",
        data=res_df,
        jitter=False,
        dodge=False,
        palette=palette,
    )
    #grid.set(yscale="log", ylim=(args.ymin, args.ymax))
    grid.set(yscale="log")
    sns.despine()
    plt.tight_layout()
    plt.savefig(args.out_plot)


if __name__ == "__main__":
    main(sys.argv[1:])
