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
import logging
import torch

from spinn2.sier_net import SierNetEstimator
####################################### parameter settings
data_augmentation = False
# num_predictions = 20
batch_size = 32 # mini batch for training
#num_classes = 3   #### categories of labels
epochs = 2      #### iterations of trainning, with GPU 1080, each epoch takes about 60s
#length_TF =3057  # number of divide data parts
# num_predictions = 20
out_model_file = '_output/jean_test_model.pt'

log_file = "_output/jean_log.txt"

seed = 0
n_layers = 5
n_hidden = 100
full_tree_pen = 0.0005
input_pen = 0.0001
num_batches = 5
max_iters = 500
max_prox_iters = 200
###################################################


def load_data_TF2(indel_list,data_path): # cell type specific  ## random samples for reactome is not enough, need borrow some from keggp
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
            xxdata_list.append(xdata[k,:,:,:])
            yydata.append(ydata[k])
        count_setx = count_setx + len(ydata)
        count_set.append(count_setx)
        print (i,len(ydata))
    yydata_array = np.array(yydata)
    yydata_x = yydata_array.astype('int')
    print (np.array(xxdata_list).shape)
    return((np.array(xxdata_list),yydata_x,count_set))


if len(sys.argv) < 4:
    print ('No enough input files')
    sys.exit()

logging.basicConfig(
    format="%(message)s", filename=log_file, level=logging.DEBUG
)

length_TF =int(sys.argv[1]) # number of data parts divided
data_path = sys.argv[2]
num_classes = int(sys.argv[3])
whole_data_TF = [i for i in range(length_TF)]
(x_train, y_train,count_set_train) = load_data_TF2(whole_data_TF,data_path)
x_train = x_train.reshape((x_train.shape[0],-1))
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
estimator = SierNetEstimator(
    n_inputs=n_inputs,
    input_filter_layer=True,
    n_layers=n_layers,
    n_hidden=n_hidden,
    n_out=num_classes,
    full_tree_pen=full_tree_pen,
    input_pen=input_pen,
    batch_size=(n_obs // num_batches + 1),
    num_classes=num_classes,
    # Weight classes by inverse of their observed ratios. Trying to balance classes
    weight=(n_obs / (num_classes * np.bincount(y_train.flatten()))
    if num_classes >= 2
    else None),
)
torch.manual_seed(seed)
estimator.fit(
    x_train, y_train, max_iters=max_iters, max_prox_iters=max_prox_iters
)
print("SUCCESS")

meta_state_dict = estimator.get_params()
meta_state_dict["state_dicts"] = [
    estimator.net.state_dict()
]
torch.save(meta_state_dict, out_model_file)
