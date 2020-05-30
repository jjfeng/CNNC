#python get_xy_label_data_cnn_combine_from_database.py data/bulk_gene_list.txt data/sc_gene_list.txt data/mmukegg_new_new_unique_rand_labelx.txt data/mmukegg_new_new_unique_rand_labelx_num_sy.txt /fh/fast/matsen_e/jfeng2/CNNC_data/mouse_bulk.h5 None 0

python get_xy_label_data_cnn_combine_from_database.py None data/sc_gene_list.txt data/bone_marrow_gene_pairs_200.txt data/bone_marrow_gene_pairs_200_num_small.txt None /fh/fast/matsen_e/jfeng2/CNNC_data/rank_total_gene_rpkm.h5 1

# Train a new model for 3 folds -- maybe you can run lots of this in parallel to do CV
python train_new_model/train_with_labels_three_foldx.py 3 NEPDF_data 3
# Train on all the data
python train_new_model/train_with_labels_wholedatax.py 3 NEPDF_data 3
# Try my new model on trained data
python predict_no_y.py  3 NEPDF_data 3 /home/jfeng2/CNNC/_output/whole_model_test/jean_test_model.h5


# create train data
python get_xy_label_data_cnn_combine_from_database.py None data/sc_gene_list.txt data/bone_marrow_gene_pairs_200.txt data/bone_marrow_gene_pairs_200_num_small.txt None /fh//fast/matsen_e/jfeng2/CNNC_data/bone_marrow_cell.h5 1 NEPDF_data

# create test data
python get_xy_label_data_cnn_combine_from_database.py None data/sc_gene_list.txt data/bone_marrow_gene_pairs_200.txt data/bone_marrow_gene_pairs_200_num_test.txt None /fh//fast/matsen_e/jfeng2/CNNC_data/bone_marrow_cell.h5 1 NEPDF_data_test

python train_with_labels_wholedatax_easiernet.py  3 NEPDF_data 3
python predict_easiernet.py 3 NEPDF_data 3 _output/jean_test_model.pt

# Trained model is binary, but the training data is for all 3 outcomes.
python predict_no_y.py  1 NEPDF_data 1 trained_models/GTRD_bone_keras_cnn_trained_model_shallow.h5
