
# create train data
python get_xy_label_data_cnn_combine_from_database.py None data/sc_gene_list.txt data/bone_marrow_gene_pairs_200.txt data/bone_marrow_gene_pairs_200_num_train.txt None /fh//fast/matsen_e/jfeng2/CNNC_data/bone_marrow_cell.h5 1 NEPDF_data

# create test data
python get_xy_label_data_cnn_combine_from_database.py None data/sc_gene_list.txt data/bone_marrow_gene_pairs_200.txt data/bone_marrow_gene_pairs_200_num_test.txt None /fh//fast/matsen_e/jfeng2/CNNC_data/bone_marrow_cell.h5 1 NEPDF_data_test

# Train CNNC on all the data
python train_with_labels_wholedatax.py --num-tf 10 --data-path NEPDF_data --num-classes 3 --out-model /home/jfeng2/CNNC/_output/whole_model_test/cnnc_test_model.h5 --epochs 3
python predict_no_y.py 1 NEPDF_data_test 3 /home/jfeng2/CNNC/_output/whole_model_test/cnnc_test_model.h5

# Train DNN on all the data
python train_with_labels_wholedatax.py --num-tf 10 --data-path NEPDF_data --num-classes 3 --out-model /home/jfeng2/CNNC/_output/whole_model_test/dnn_test_model.h5 --epochs 3 --fit-dnn
python predict_no_y.py 1 NEPDF_data_test 3 /home/jfeng2/CNNC/_output/whole_model_test/dnn_test_model.h5

# Train easier net
python train_with_labels_wholedatax_easiernet.py --num-tf 10 --data-path NEPDF_data --num-classes 3 --out-model _output/jean_test_model.pt  --num-inits 10 --max-iters 31 --max-prox-iters 11
python predict_easiernet.py 1 NEPDF_data_test 3 _output/jean_test_model.pt
