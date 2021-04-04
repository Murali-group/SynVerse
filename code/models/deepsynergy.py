import os, sys

import pandas as pd
import numpy as np
import pickle
import gzip
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0" #specify GPU
import keras as K
import tensorflow as tf
# from keras import backend
from tensorflow.python.keras import backend
from tensorflow.python.keras.backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# import utils
from models.utils import create_drug_drug_pairs_feature, create_syn_non_syn_feat_labels,save_drug_syn_probability

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


config = tf.compat.v1.ConfigProto(
         allow_soft_placement=True,
         gpu_options = tf.compat.v1.GPUOptions(allow_growth=True))
set_session(tf.compat.v1.Session(config=config))


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def create_model(input_shape, layer_setup, lr, input_dropout, dropout, act_func):

    inputs = keras.Input(shape=(input_shape,), name="features", sparse=True)
    for i in range(len(layer_setup)):
        if i == 0:
            x1 = layers.Dense(layer_setup[i], activation=act_func, name="dense_"+str(i))(inputs)
            x = layers.Dropout(rate = input_dropout)(x1)
        elif i == len(layer_setup)-1:
            outputs = layers.Dense(layer_setup[i], activation="sigmoid", name="predictions")(x)
        else:
            x1 = layers.Dense(layer_setup[i], activation=act_func, name="dense_"+str(i))(x)
            x = layers.Dropout(rate = dropout)(x1)


    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),  # Optimizer
        # Loss function to minimize
        loss='binary_crossentropy',

    )

    return model


# def run_deepsynergy_model(drug_maccs_keys_targets_feature_df,\
#                         synergy_df,test_synergy_df, non_synergy_df, cross_validation_folds,neg_cross_validation_folds, run_,out_dir, config_map):

def run_deepsynergy_model(drug_maccs_keys_targets_feature_df, \
                              synergy_df, non_synergy_df, cross_validation_folds,
                              neg_cross_validation_folds, run_, out_dir, config_map):

    ## config parameters
    number_of_folds = config_map['ml_models_settings']['cross_val']['folds']
    ds_settings = config_map['ml_models_settings']['algs']['deepsynergy']
    layer_setups = ds_settings['layers']

    in_hid_dropouts = ds_settings['in_hid_dropouts']
    # dropouts = ds_settings['dropout']
    # input_dropouts = ds_settings['input_dropout']
    lrs = ds_settings['lr']
    act_func = ds_settings['act_func']
    epochs = ds_settings['epochs']
    batch_size = ds_settings['batch_size']

    cell_lines = synergy_df['Cell_line'].unique()
    cell_line_2_idx = {cell_line: i for i, cell_line in enumerate(cell_lines)}
    synergy_df['cell_line_idx'] = synergy_df['Cell_line'].apply(lambda x: cell_line_2_idx[x])
    non_synergy_df['cell_line_idx'] = non_synergy_df['Cell_line'].apply(lambda x: cell_line_2_idx[x])
    # test_synergy_df['cell_line_idx'] = test_synergy_df['Cell_line'].apply(lambda  x: cell_line_2_idx[x])
    #one_hot_encoding of cell line feature
    cell_line_feat_df = pd.DataFrame(np.eye(len(cell_lines), dtype=int))

    epoch_to_optimize_file = out_dir + 'epochs_to_optimize.txt'
    os.makedirs(os.path.dirname(epoch_to_optimize_file), exist_ok=True)
    epoch_to_optimize = open(epoch_to_optimize_file, 'w')


    count=0
    for layer_setup in layer_setups:
        for lr in lrs:
            for in_hid_dropout in in_hid_dropouts:
                epoch_to_optimize = open(out_dir + 'epochs_to_optimize.txt', 'a')
                input_dropout = in_hid_dropout[0]
                dropout = in_hid_dropout[1]

                predictions_dict = {'drug_1': [], 'drug_2': [], 'cell_line': [], 'predicted': [],
                                    'true': [], 'val_fold': []}
                # predictions_test_dict = {fold: [] for fold in range(number_of_folds)}
                for fold in range(number_of_folds):

                    #########################  prepare feature and labels  ############################
                    training_synergy_df = synergy_df[~synergy_df.index.isin(cross_validation_folds[fold])] \
                        [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

                    training_non_synergy_df = non_synergy_df[~non_synergy_df.index.isin(neg_cross_validation_folds[fold])] \
                        [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

                    validation_synergy_df = synergy_df[synergy_df.index.isin(cross_validation_folds[fold])] \
                        [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

                    validation_non_synergy_df = non_synergy_df[non_synergy_df.index.isin(neg_cross_validation_folds[fold])] \
                        [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

                    train_feat_df, train_label = create_syn_non_syn_feat_labels(drug_maccs_keys_targets_feature_df, cell_line_feat_df, \
                                                                             training_synergy_df, training_non_synergy_df)

                    val_feat_df, val_label = create_syn_non_syn_feat_labels(drug_maccs_keys_targets_feature_df, cell_line_feat_df, \
                                                                         validation_synergy_df, validation_non_synergy_df)

                    train_feat = csr_matrix(train_feat_df.values)
                    val_feat = csr_matrix(val_feat_df.values)
                    # test_feat = csr_matrix(create_drug_drug_pairs_feature(drug_maccs_keys_targets_feature_df,\
                    #                                                       cell_line_feat_df,test_synergy_df).values)

                    ##########################  train model  ##########################################
                    best_model_file = out_dir + 'run_' + str(run_) + '/'+'best_model_layers_'+str(layer_setup)+'_e_'+\
                        str(epochs) +'_lr_'+str(lr) +'_batch_'+ str(batch_size) +'_dr_'+\
                        str(input_dropout)+'_'+str(dropout)+'_act_'+str(act_func)+'fold_' + str(fold)+'.h5'
                    os.makedirs(os.path.dirname(best_model_file), exist_ok=True)


                    es = EarlyStopping(monitor='val_loss', min_delta= 0.00001, mode='min', verbose=0, patience=50)
                    mc = ModelCheckpoint(best_model_file, monitor='val_loss', mode='min', verbose=0,
                                         save_best_only=True)

                    print('input shape: ',train_feat.shape[1])
                    model = create_model(train_feat.shape[1], layer_setup, lr, input_dropout, dropout, act_func)
                    hist = model.fit(train_feat, train_label, epochs=epochs, callbacks=[es, mc], shuffle=True,\
                                     batch_size=batch_size, validation_data=(val_feat, val_label), verbose=1)
                    epoch_to_optimize.write('\n Model: '+str(count) + ', epochs: ' + str(len(hist.history['loss'])))

                    # val_loss = hist.history['val_loss']

                    #free memory
                    backend.clear_session()

                    best_model = load_model(best_model_file)
                    # eval_result = best_model.evaluate(val_feat,val_label, batch_size=128)
                    # print(eval_result)

                    predictions = best_model.predict(val_feat)
                    print('predictions: ', type(predictions), predictions.shape)

                    # predictions_test_dict[fold] = list(best_model.predict(test_feat).squeeze(axis=1))
                    row=0
                    for drug_1, drug_2, cell_line in val_feat_df.index:
                        predictions_dict['drug_1'].append(drug_1)
                        predictions_dict['drug_2'].append(drug_2)
                        predictions_dict['cell_line'].append(cell_line)
                        predictions_dict['predicted'].append(predictions[row][0])
                        predictions_dict['true'].append(val_label[row])
                        predictions_dict['val_fold'].append(fold)

                        row+=1

                    # free memory
                    del train_feat
                    del val_feat
                    # del test_feat

                predictions_df = pd.DataFrame.from_dict(predictions_dict)
                pos_df = predictions_df[predictions_df['true'] == 1]
                neg_df = predictions_df[predictions_df['true'] == 0]


                save_drug_syn_probability(pos_df, neg_df, layer_setup, lr, input_dropout, dropout, batch_size, \
                                          epochs, act_func, run_, out_dir)

                # predictions_test_df = pd.DataFrame.from_dict(predictions_test_dict)
                # predictions_test_df.to_csv(out_dir+str(count)+'.tsv', sep='\t')
                count+=1
                epoch_to_optimize.close()