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

from train_with_labels_wholedatax_easiernet import load_data_TF2


################################
#num_classes = 3 ################################# the number of classes might vary for your special application
#length_TF =2 # number of data parts divided
#def load_data_TF2(indel_list,data_path): # cell type specific  ## random samples for reactome is not enough, need borrow some from keggp
#    """
#    This method does not load y labels
#    """
#    import random
#    import numpy as np
#    xxdata_list = []
#    yydata = []
#    count_set = [0]
#    count_setx = 0
#    for i in indel_list:#len(h_tf_sc)):
#        xdata = np.load(data_path+'/Nxdata_tf' + str(i) + '.npy')
#        for k in range(xdata.shape[0]):
#            xxdata_list.append(xdata[k,:,:,:])
#        count_setx = count_setx + xdata.shape[0]
#        count_set.append(count_setx)
#    return((np.array(xxdata_list),count_set))

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

save_dir = os.path.join(os.getcwd(),'predict_results_no_y_1')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
print ('do predict')
y_log_prob = model.predict_log_proba(x_test)
y_prob = np.exp(y_log_prob)
loss_fn = torch.nn.CrossEntropyLoss()
print(y_test.dtype)
test_loss = loss_fn(torch.tensor(y_log_prob), torch.tensor(y_test[:,0], dtype=torch.long))

y_discrete_pred = np.argmax(y_prob, axis=1).reshape((-1,1))
test_acc = np.mean(y_test == y_discrete_pred)
print("TEST ACC", test_acc)
print("TEST loss", test_loss)
#np.save(save_dir+'/y_test.npy',y_test)
np.save(save_dir+'/y_predict.npy',y_prob)
s = open (save_dir+'/gene_index.txt','w')
for i in count_set:
    s.write(str(i)+'\n')
s.close()
######################################
