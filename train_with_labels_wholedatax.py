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
from networks import make_dnn
from train_with_labels_wholedatax_easiernet import load_data_TF2

def main(args=sys.argv[1:]):
    ####################################### parameter settings
    seed = 0
    tf.random.set_seed(0)
    data_augmentation = False
    # num_predictions = 20
    batch_size = 32 # mini batch for training
    #num_classes = 3   #### categories of labels
    epochs = 200      #### iterations of trainning, with GPU 1080, each epoch takes about 60s
    #length_TF =3057  # number of divide data parts
    # num_predictions = 20
    model_name = 'dnn_test_model.h5'
    fit_dnn = True
    ###################################################


    if len(sys.argv) < 4:
        print ('No enough input files')
        sys.exit()
    length_TF =int(sys.argv[1]) # number of data parts divided
    data_path = sys.argv[2]
    num_classes = int(sys.argv[3])
    whole_data_TF = [i for i in range(length_TF)]
    (x_train, y_train,count_set_train) = load_data_TF2(whole_data_TF,data_path, flatten=fit_dnn)
    print(x_train.shape, 'x_train samples')
    save_dir = os.path.join(os.getcwd(), '_output', 'whole_model_test')
    if num_classes > 2:
        y_train = keras.utils.to_categorical(y_train, num_classes)
    print(y_train.shape, 'y_train samples')
        ###########
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        ############
    model = make_dnn(x_train, num_classes, num_hidden=50, dropout_rate=0.15)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=300, verbose=0, mode='auto')
    checkpoint1 = ModelCheckpoint(filepath=save_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    checkpoint2 = ModelCheckpoint(filepath=save_dir + '/weights.hdf5', monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1)
    callbacks_list = [checkpoint2, early_stopping]
    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_split=0.2,shuffle=True, callbacks=callbacks_list)

        # Save model and weights

    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
        # Score trained model.
    ############################################################################## plot training process
    plt.figure(figsize=(10, 6))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    plt.savefig(save_dir+'/end_result.pdf')

if __name__ == "__main__":
    main(sys.argv[1:])
