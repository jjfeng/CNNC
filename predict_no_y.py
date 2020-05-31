# Usage: python predict_no_y.py  number_of_separation NEPDF_pathway model_pathway
# command line in developer's linux machine :
# python predict_no_y.py  9 /home/yey3/cnn_project/code3/NEPDF_data   /home/yey3/cnn_project/code3/trained_model/models/KEGG_keras_cnn_trained_model_shallow2.h5
from __future__ import print_function
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
from networks import make_dnn, make_cnnc
from common import get_perf
from train_with_labels_wholedatax_easiernet import load_data_TF2


fit_dnn = False
length_TF =int(sys.argv[1]) # number of data parts divided
data_path = sys.argv[2]
num_classes = int(sys.argv[3])
model_path = sys.argv[4] ## KEGG or Reactome or TF
print("MODEL PATH", model_path)
print ('select', type)
whole_data_TF = [i for i in range(length_TF)]
test_TF = [i for i in range (length_TF)]
(x_test, y_test, count_set) = load_data_TF2(test_TF,data_path, flatten=fit_dnn)
print(x_test.shape, 'x_test samples')
############

if fit_dnn:
    model = make_dnn(x_test, num_classes, num_hidden=50, dropout_rate=0.15)
else:
    model = make_cnnc(x_test, num_classes)
model.load_weights(model_path)
###########################################################
print ('predict')
y_predict = model.predict(x_test)
perf_dict = get_perf(y_predict, y_test)
print(perf_dict)
