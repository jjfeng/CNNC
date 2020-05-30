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
import torch
from spinn2.network import SierNet

from common import get_perf
from train_with_labels_wholedatax_easiernet import load_data_TF2


def load_easier_net(model_file):
    meta_state_dict = torch.load(model_file)
    model = SierNet(
        n_layers=meta_state_dict["n_layers"],
        n_input=meta_state_dict["n_inputs"],
        n_hidden=meta_state_dict["n_hidden"],
        n_out=meta_state_dict["n_out"],
        input_filter_layer=meta_state_dict["input_filter_layer"],
    )
    model.load_state_dict(meta_state_dict["state_dicts"][0])
    model.eval()
    return model

length_TF =int(sys.argv[1]) # number of data parts divided
data_path = sys.argv[2]
num_classes = int(sys.argv[3])
model_path = sys.argv[4] ## KEGG or Reactome or TF
print("MODEL PATH", model_path)
print ('select', type)
whole_data_TF = [i for i in range(length_TF)]
test_TF = [i for i in range (length_TF)]
(x_test, y_test, count_set) = load_data_TF2(test_TF,data_path)
x_test = x_test.reshape((x_test.shape[0],-1))
print(x_test.shape, 'x_test samples')
y_test = y_test.reshape((y_test.size, 1))
############

model = load_easier_net(model_path)

y_log_prob = model.predict_log_proba(x_test)
y_predict = np.exp(y_log_prob)
perf_dict = get_perf(y_predict, y_test)
print(perf_dict)
