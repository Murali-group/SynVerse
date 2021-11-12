import os, sys

import pandas as pd
import numpy as np
import pickle
import gzip
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0" #specify GPU
# import keras as K
# import tensorflow as tf
# # from keras import backend
# from tensorflow.python.keras import backend
from tensorflow.python.keras.backend import set_session
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
from keras import backend
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
# from numba import cuda
import datetime

# import utils
from models.utils import create_drug_drug_pairs_feature, create_syn_non_syn_feat_labels

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = set_session(tf.compat.v1.Session(config=config))


def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    if feat_filt is None:
        feat_filt = std1!=0
    X = X[:,feat_filt]
    X = np.ascontiguousarray(X)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    X = (X-means1)/std1[feat_filt]
    if norm == 'norm':
        return(X, means1, std1, feat_filt)
    elif norm == 'tanh':
        return(np.tanh(X), means1, std1, feat_filt)
    elif norm == 'tanh_norm':
        X = np.tanh(X)
        if means2 is None:
            means2 = np.mean(X, axis=0)
        if std2 is None:
            std2 = np.std(X, axis=0)
        X = (X-means2)/std2
        X[:,std2==0]=0
        return(X, means1, std1, means2, std2, feat_filt)


def save_drug_syn_probability(pos_df, neg_df, layer_setup, lr, input_dropout, dropout,norm, batch_size, epochs, act_func,
                              use_genex, use_target, reduce_dim, out_dir):
    # inputs: df with link prediction probability after applying sigmoid on models predicted score for an edges(positive, negative)
    pos_out_file = out_dir + \
                   '/pos_val_scores' + '_layers_' + str(layer_setup) + '_e_' + str(epochs) + '_lr_' + str(
        lr) + '_batch_' + str(batch_size) + '_dr_' + \
                   str(input_dropout) + '_' + str(dropout) + '_norm_'+norm+'_act_' + str(act_func) +'_use_genex_' + str(use_genex)+\
                   '_use_target_' + str(use_target)+'_reduce_dim_' + str(reduce_dim) + '.tsv'

    neg_out_file = out_dir + \
                   '/neg_val_scores' + '_layers_' + str(layer_setup) + '_e_' + str(epochs) + '_lr_' + str(
        lr) + '_batch_' + str(batch_size) + '_dr_' + \
                   str(input_dropout) + '_' + str(dropout) +'_norm_'+norm+ '_act_' + str(act_func)+'_use_genex_' + str(use_genex) +\
                   '_use_target_' + str(use_target)+'_reduce_dim_' + str(reduce_dim) +'.tsv'

    os.makedirs(os.path.dirname(pos_out_file), exist_ok=True)
    os.makedirs(os.path.dirname(neg_out_file), exist_ok=True)

    pos_df.to_csv(pos_out_file, sep='\t')
    neg_df.to_csv(neg_out_file, sep='\t')

def save_val_eval (val_loss, layer_setup, lr, input_dropout, dropout,norm, batch_size, \
                              epochs, act_func,use_genex, use_target,reduce_dim, out_dir, fold_no ):
    model_info = 'layers_' + str(layer_setup) + '_e_' + str(epochs) + '_lr_' + str(
        lr) + '_batch_' + str(batch_size) + '_dr_' + \
                   str(input_dropout) + '_' + str(dropout) + '_norm_'+norm+'_act_' + str(act_func) + '_use_genex_' + str(use_genex) + \
                   '_use_target_' + str(use_target)+'_reduce_dim_' + str(reduce_dim) +'_'

    val_file = out_dir + model_info + 'model_val_loss.txt'
    os.makedirs(os.path.dirname(val_file), exist_ok=True)
    if fold_no==0:
        val_file  = open(val_file, 'w')
    else:
        val_file = open(val_file, 'a')
    # val_file.write(model_info)
    val_file.write('val_loss: '+str(val_loss))
    val_file.write('\n\n')
    val_file.close()

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

def run_deepsynergy_model(drug_maccs_keys_targets_feature_df,drug_maccs_keys_feature_df,\
                            gene_expression_feature_df,
                              synergy_df, non_synergy_df,cell_line_2_idx,idx_2_cell_line,
                              cross_validation_folds,
                              neg_cross_validation_folds,
                              ds_params, out_dir, config_map):
    '''
    cross_validation_folds is in the form:
    cross_validation_folds = {i: {'test': [], 'train': [], 'val': []} for i in range(number_of_folds)}
    '''


    ## config parameters
    number_of_folds = config_map['ml_models_settings']['cross_val']['folds']
    cell_lines = synergy_df['Cell_line'].unique()
    synergy_df['cell_line_idx'] = synergy_df['Cell_line'].apply(lambda x: cell_line_2_idx[x])
    non_synergy_df['cell_line_idx'] = non_synergy_df['Cell_line'].apply(lambda x: cell_line_2_idx[x])
    use_genex = ds_params['use_genex']
    use_target = ds_params['use_target']
    reduce_dim = ds_params['reduce_dim']

    if use_genex:
        # gene expression feat for cell lines
        cell_line_feat_df = gene_expression_feature_df
        # cell_line_feat_df['cell_line_idx'] =  cell_line_feat_df.index.apply(lambda x: cell_line_2_idx)
        # cell_line_feat_df.set_index('cell_line_idx', drop=True, inplace= True)

    else:
        #one_hot_encoding of cell line feature
        cell_line_feat_df = pd.DataFrame(np.eye(len(cell_lines), dtype=int))


    epoch_to_optimize_file = out_dir + 'epochs_to_optimize.txt'
    os.makedirs(os.path.dirname(epoch_to_optimize_file), exist_ok=True)
    epoch_to_optimize = open(epoch_to_optimize_file, 'a')

    #model parameters
    count=0
    layers = ds_params['layers']
    lr = ds_params['lr']
    epochs = ds_params['e']
    act_func = ds_params['act']
    batch_size = ds_params['batch']
    norm = ds_params['norm']
    input_dropout = ds_params['dr'][0]
    dropout = ds_params['dr'][1]


    test_predictions_dict = {'drug_1': [], 'drug_2': [], 'cell_line': [], 'predicted': [],
                        'true': []}
    val_predictions_dict = {'drug_1': [], 'drug_2': [], 'cell_line': [], 'predicted': [],
                             'true': []}

    # predictions_test_dict = {fold: [] for fold in range(number_of_folds)}


    for fold in range(number_of_folds):


        #########################  prepare feature and labels  ############################
        training_synergy_df = synergy_df[synergy_df.index.isin(cross_validation_folds[fold]['train'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

        training_non_synergy_df = non_synergy_df[non_synergy_df.index.isin(neg_cross_validation_folds[fold]['train'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

        validation_synergy_df = synergy_df[synergy_df.index.isin(cross_validation_folds[fold]['val'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

        validation_non_synergy_df = non_synergy_df[non_synergy_df.index.isin(neg_cross_validation_folds[fold]['val'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

        validation_es_synergy_df = synergy_df[synergy_df.index.isin(cross_validation_folds[fold]['val_es'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

        validation_es_non_synergy_df = non_synergy_df[non_synergy_df.index.isin(neg_cross_validation_folds[fold]['val_es'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

        test_synergy_df = synergy_df[synergy_df.index.isin(cross_validation_folds[fold]['test'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

        test_non_synergy_df = non_synergy_df[non_synergy_df.index.isin(neg_cross_validation_folds[fold]['test'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]


        if use_target:

            train_feat_df, train_label = create_syn_non_syn_feat_labels(drug_maccs_keys_targets_feature_df, cell_line_feat_df, \
                                                                     training_synergy_df, training_non_synergy_df)

            val_feat_df, val_label = create_syn_non_syn_feat_labels(drug_maccs_keys_targets_feature_df, cell_line_feat_df, \
                                                                 validation_synergy_df, validation_non_synergy_df)

            val_es_feat_df, val_es_label = create_syn_non_syn_feat_labels(drug_maccs_keys_targets_feature_df,
                                                                    cell_line_feat_df, \
                                                                    validation_es_synergy_df, validation_es_non_synergy_df)
            test_feat_df, test_label = create_syn_non_syn_feat_labels(drug_maccs_keys_targets_feature_df,
                                                                      cell_line_feat_df,
                                                                      test_synergy_df, test_non_synergy_df)
        else:
            train_feat_df, train_label = create_syn_non_syn_feat_labels(drug_maccs_keys_feature_df,
                                                                        cell_line_feat_df, \
                                                                        training_synergy_df, training_non_synergy_df)

            val_feat_df, val_label = create_syn_non_syn_feat_labels(drug_maccs_keys_feature_df,
                                                                    cell_line_feat_df, \
                                                                    validation_synergy_df, validation_non_synergy_df)
            val_es_feat_df, val_es_label = create_syn_non_syn_feat_labels(drug_maccs_keys_feature_df,
                                                                    cell_line_feat_df, \
                                                                    validation_es_synergy_df, validation_es_non_synergy_df)
            test_feat_df, test_label = create_syn_non_syn_feat_labels(drug_maccs_keys_feature_df,
                                                                      cell_line_feat_df,
                                                                      test_synergy_df, test_non_synergy_df)

        #normalize data

        if norm == "tanh_norm":
            train_feat, mean, std, mean2, std2, feat_filt = normalize(train_feat_df.values, norm=norm)
            val_es_feat, mean_vs, std_vs, mean2_vs, std2_vs, feat_filt_vs = normalize(val_es_feat_df.values, mean, std, mean2, std2,
                                                                    feat_filt=feat_filt, norm=norm)
            val_feat, mean_v, std_v, mean2_v, std2_v, feat_filt_v = normalize(val_feat_df.values, mean, std, mean2, std2,
                                                                 feat_filt=feat_filt, norm=norm)
            test_feat,mean_t, std_t, mean2_t, std2_t, feat_filt_t = normalize(test_feat_df.values, mean, std, mean2, std2,
                                                                 feat_filt=feat_filt, norm=norm)
        elif norm in ['tanh', 'norm']:
            train_feat, mean, std, feat_filt = normalize(train_feat_df.values, norm=norm)
            val_es_feat, mean_vs, std_vs, feat_filt_vs = normalize(val_es_feat_df.values, mean, std, feat_filt=feat_filt, norm=norm)
            val_feat, mean_v, std_v, feat_filt_v = normalize(val_feat_df.values, mean, std, feat_filt=feat_filt, norm=norm)
            test_feat, mean_t, std_t, feat_filt_t = normalize(test_feat_df.values, mean, std,feat_filt=feat_filt, norm=norm)
        elif norm=='no':
            train_feat = train_feat_df.values
            val_es_feat = val_es_feat_df.values
            val_feat = val_feat_df.values
            test_feat = test_feat_df.values
        #
        # train_feat = csr_matrix(train_feat)
        # val_feat = csr_matrix(val_feat)
        # val_es_feat = csr_matrix(val_es_feat)
        # test_feat = csr_matrix(test_feat)


        ##########################  train model  ##########################################

        best_model_file = out_dir +'best_model_layers_'+str(layers)+'_e_'+\
            str(epochs) +'_lr_'+str(lr) +'_batch_'+ str(batch_size) +'_dr_'+\
            str(input_dropout)+'_'+str(dropout)+'_act_'+str(act_func)+'_use_genex_' + str(use_genex)+\
                          '_use_target_' + str(use_target)+'_fold_' + str(fold)+'.h5'
        os.makedirs(os.path.dirname(best_model_file), exist_ok=True)


        es = EarlyStopping(monitor='val_loss', min_delta= 0.00001, mode='min', verbose=0, patience=50)
        mc = ModelCheckpoint(best_model_file, monitor='val_loss', mode='min', verbose=0,
                             save_best_only=True)
        #
        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # print('log dir: ', log_dir)
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        print('input shape: ',train_feat.shape[1])
        model = create_model(train_feat.shape[1], layers, lr, input_dropout, dropout, act_func)
        hist = model.fit(train_feat, train_label, epochs=epochs, callbacks=[es, mc], shuffle=True,\
                         batch_size=batch_size, validation_data=(val_es_feat, val_es_label), verbose=1)
        epoch_to_optimize.write('\n Model: '+str(count) + ', epochs: ' + str(len(hist.history['loss'])))

        # val_loss = hist.history['val_loss']

        #free memory
        backend.clear_session()

        del model
        best_model = load_model(best_model_file)


        #prediction on val
        val_loss = best_model.evaluate(val_feat, val_label, batch_size=128)
        save_val_eval(val_loss, layers, lr, input_dropout, dropout, norm, batch_size, \
                      epochs, act_func, use_genex, use_target,reduce_dim, out_dir, fold)

        val_predictions = best_model.predict(val_feat)
        row = 0
        for drug_1, drug_2, cell_line_idx in val_feat_df.index:
            val_predictions_dict['drug_1'].append(drug_1)
            val_predictions_dict['drug_2'].append(drug_2)
            val_predictions_dict['cell_line'].append(idx_2_cell_line[cell_line_idx])
            val_predictions_dict['predicted'].append(val_predictions[row][0])
            val_predictions_dict['true'].append(val_label[row])
            # predictions_dict['fold'].append(fold)

            row += 1


        #prediction on test
        test_predictions = best_model.predict(test_feat)
        row=0
        for drug_1, drug_2, cell_line_idx in test_feat_df.index:
            test_predictions_dict['drug_1'].append(drug_1)
            test_predictions_dict['drug_2'].append(drug_2)
            test_predictions_dict['cell_line'].append(idx_2_cell_line[cell_line_idx])
            test_predictions_dict['predicted'].append(test_predictions[row][0])
            test_predictions_dict['true'].append(test_label[row])
            # predictions_dict['fold'].append(fold)

            row+=1

        # free memory
        del train_feat
        del val_feat
        del val_es_feat
        del test_feat
        del best_model
        os.remove(best_model_file)
        # device = cuda.get_current_device()
        # device.reset()

    # val_prediction_df = pd.DataFrame.from_dict(val_predictions_dict)

    test_predictions_df = pd.DataFrame.from_dict(test_predictions_dict)
    test_pos_df = test_predictions_df[test_predictions_df['true'] == 1]
    test_neg_df = test_predictions_df[test_predictions_df['true'] == 0]


    save_drug_syn_probability(test_pos_df, test_neg_df, layers, lr, input_dropout, dropout,norm, batch_size, \
                              epochs, act_func,use_genex, use_target, reduce_dim, out_dir)


    # predictions_test_df = pd.DataFrame.from_dict(predictions_test_dict)
    # predictions_test_df.to_csv(out_dir+str(count)+'.tsv', sep='\t')
    count+=1
    epoch_to_optimize.close()