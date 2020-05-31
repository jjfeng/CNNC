from __future__ import print_function
# Usage  python train_with_labels_wholedata.py number_of_data_parts_divided
# command line in developer's linux machine :
# module load cuda-8.0 using GPU
#srun -p gpu --gres=gpu:1 -c 2 --mem=20Gb python train_with_labels_wholedatax.py 9 /home/yey3/cnn_project/code3/NEPDF_data 3 > results_whole.txt
#######################OUTPUT
# it will generate a folder 'wholeXXXXX', in which 'keras_cnn_trained_model_shallow.h5' is the final trained model
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint
import os,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
####################
# Jean modifications
import argparse
from networks import make_dnn, make_cnnc
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
    parser.add_argument("--data-file", type=str, default="_output/data.npz")
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
        "--do-binary", action="store_true", default=False, help="fit binary outcome"
    )
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
        "--batch-size", type=int, default=32
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
        "--epochs", type=int, default=40, help="Number of Adam epochs"
    )
    parser.add_argument("--log-file", type=str, default="_output/log_nn.txt")
    parser.add_argument("--out-model-file", type=str, default="_output/nn.pt")
    args = parser.parse_args()

    assert args.num_classes != 1

    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)
    tf.random.set_seed(args.seed)

    whole_data_TF = [i for i in range(args.num_tf) if i != args.exclude_tf]
    (x_train, y_train,count_set_train) = load_data_TF2(whole_data_TF,args.data_path,binary_outcome=args.do_binary, flatten=args.fit_dnn)
    print(x_train.shape, 'x_train samples')
    save_dir = args.out_model_file.replace(".h5", "_scratch")
    print(save_dir)
    if args.num_classes > 2:
        y_train = keras.utils.to_categorical(y_train, args.num_classes)
    print(y_train.shape, 'y_train samples')
        ###########
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        ############
    if args.fit_dnn:
        model = make_dnn(x_train, args.num_classes, num_layers=args.n_layers, num_hidden=args.n_hidden, dropout_rate=args.dropout_rate)
    else:
        model = make_cnnc(x_train, args.num_classes)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=300, verbose=0, mode='auto')
    checkpoint1 = ModelCheckpoint(filepath=save_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    checkpoint2 = ModelCheckpoint(filepath=save_dir + '/weights.hdf5', monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1)
    callbacks_list = [checkpoint2, early_stopping]

    history = model.fit(x_train, y_train,batch_size=args.batch_size,epochs=args.epochs,validation_split=0.2,shuffle=True, callbacks=callbacks_list)

    # Save model and weights
    model.save(args.out_model_file)

    ############################################################################## plot training process
    #plt.figure(figsize=(10, 6))
    #plt.subplot(1,2,1)
    #plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    #plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.grid()
    #plt.legend(['train', 'val'], loc='upper left')
    #plt.subplot(1,2,2)
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('model loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'val'], loc='upper left')
    #plt.grid()
    #plt.savefig(save_dir+'/end_result.pdf')

if __name__ == "__main__":
    main(sys.argv[1:])
