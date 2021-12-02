import pandas as pd
import numpy as np
import os
import nvidia_smi

def get_drug_feat(use_target, use_maccs, drug_maccs_keys_targets_feature_df):
    if (use_maccs == True) & (use_target == True) :
        drug_feat_df = drug_maccs_keys_targets_feature_df

    elif (use_maccs == True) & (use_target == False):
        drug_feat_df = drug_maccs_keys_targets_feature_df.iloc[:,1:167:1]  #just maccskeys
        drug_feat_df['pubchem_cid'] = list(drug_maccs_keys_targets_feature_df['pubchem_cid'])

    elif (use_maccs == False) & (use_target == True):
        drug_feat_df = drug_maccs_keys_targets_feature_df.iloc[:, 167::1]
        drug_feat_df['pubchem_cid'] = list(drug_maccs_keys_targets_feature_df['pubchem_cid'])

    elif (use_maccs == False) & (use_target == False):
        drug_feat_df = pd.DataFrame(np.eye(len(drug_maccs_keys_targets_feature_df), dtype=int))
        drug_feat_df['pubchem_cid'] = list(drug_maccs_keys_targets_feature_df['pubchem_cid'])
        # drug_feat_df.set_index('pubchem_cid' , inplace=True)
    print(drug_feat_df.head(10))
    return drug_feat_df

def create_drug_drug_pairs_feature(drug_maccs_keys_targets_feature_df, cell_line_feat_df, synergy_df):
    drugs_cell_lines = pd.DataFrame()

    if len(synergy_df)!=0:
        drug_maccs_keys_targets_feature_df = drug_maccs_keys_targets_feature_df.set_index('pubchem_cid')

        drug_1_maccs_keys_targets = drug_maccs_keys_targets_feature_df.reindex(
            synergy_df['Drug1_pubchem_cid']).reset_index()
        drug_2_maccs_keys_targets = drug_maccs_keys_targets_feature_df.reindex(
            synergy_df['Drug2_pubchem_cid']).reset_index()
        cell_lines = cell_line_feat_df.reindex(synergy_df['cell_line_idx']).reset_index()

        drug_1_2_maccs_keys_targets_cell_lines = pd.concat(
            [drug_1_maccs_keys_targets, drug_2_maccs_keys_targets, cell_lines], axis=1)
        drug_1_2_maccs_keys_targets_cell_lines = drug_1_2_maccs_keys_targets_cell_lines.\
            set_index(['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'cell_line_idx'])
        drug_2_1_maccs_keys_targets_cell_lines = pd.concat(
            [drug_2_maccs_keys_targets, drug_1_maccs_keys_targets, cell_lines], axis=1). \
            set_index(['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'cell_line_idx'])

        # rename the columns before concatenation
        col_names_1 = list(drug_1_2_maccs_keys_targets_cell_lines.columns)
        col_names_2 = list(drug_2_1_maccs_keys_targets_cell_lines.columns)

        new_cols_1 = dict(zip(col_names_1, list(range(len(col_names_1)))))
        new_cols_2 = dict(zip(col_names_2, list(range(len(col_names_2)))))

        drug_1_2_maccs_keys_targets_cell_lines.rename(columns=new_cols_1, inplace=True)
        drug_2_1_maccs_keys_targets_cell_lines.rename(columns=new_cols_2, inplace=True)

        # print(len(drug_1_2_maccs_keys_targets_cell_lines.columns), len(drug_2_1_maccs_keys_targets_cell_lines.columns))

        drugs_cell_lines = pd.concat([drug_1_2_maccs_keys_targets_cell_lines, drug_2_1_maccs_keys_targets_cell_lines], \
                                     axis=0)

    return drugs_cell_lines


def create_syn_non_syn_feat_labels(drug_feature_df, cell_line_feat_df, \
                                   synergy_df, non_synergy_df):
    # retun 1 df and 1 numpy arrays:
    # 1. feat_df = contains feature for both postive and negative drug pairs in both drugA-drugB and drugB-drugA order
    # 2. labels: contains 0/1 label for each drug-drug-cell_line triplets.

    feat_synergy_pairs_df = create_drug_drug_pairs_feature(drug_feature_df, cell_line_feat_df, \
                                                           synergy_df)
    label_1 = np.ones(len(feat_synergy_pairs_df))
    feat_non_synergy_pairs_df = create_drug_drug_pairs_feature(drug_feature_df, cell_line_feat_df, \
                                                               non_synergy_df)
    label_0 = np.zeros(len(feat_non_synergy_pairs_df))

    feat_df = pd.concat([feat_synergy_pairs_df, feat_non_synergy_pairs_df], axis=0)

    # convert to numpy array
    # feat = feat_df.values
    label = np.concatenate((label_1, label_0), axis=0)
    return feat_df, label


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


# def save_drug_syn_probability(pos_df, neg_df, params, out_dir):
#     # inputs: df with link prediction probability after applying sigmoid on models predicted score for an edges(positive, negative)
#     pos_out_file = out_dir + \
#                    '/pos_val_scores' + + '.tsv'
#
#     neg_out_file = out_dir + \
#                    '/neg_val_scores' + '_layers_' + str(layer_setup) + '_e_' + str(epochs) + '_lr_' + str(
#         lr) + '_batch_' + str(batch_size) + '_dr_' + \
#                    str(input_dropout) + '_' + str(dropout) +'_norm_'+norm+ '_act_' + str(act_func)+'_use_genex_' + str(use_genex) +\
#                    '_use_target_' + str(use_target)+'_reduce_dim_' + str(reduce_dim) +'.tsv'
#
#     os.makedirs(os.path.dirname(pos_out_file), exist_ok=True)
#     os.makedirs(os.path.dirname(neg_out_file), exist_ok=True)
#
#     pos_df.to_csv(pos_out_file, sep='\t')
#     neg_df.to_csv(neg_out_file, sep='\t')
#
# def save_val_eval (val_loss, params, out_dir, fold_no ):
#     model_info = 'layers_' + str(layer_setup) + '_e_' + str(epochs) + '_lr_' + str(
#         lr) + '_batch_' + str(batch_size) + '_dr_' + \
#                    str(input_dropout) + '_' + str(dropout) + '_norm_'+norm+'_act_' + str(act_func) + '_use_genex_' + str(use_genex) + \
#                    '_use_target_' + str(use_target)+'_reduce_dim_' + str(reduce_dim) +'_'
#
#     val_file = out_dir + model_info + 'model_val_loss.txt'
#     os.makedirs(os.path.dirname(val_file), exist_ok=True)
#     if fold_no==0:
#         val_file  = open(val_file, 'w')
#     else:
#         val_file = open(val_file, 'a')
#     # val_file.write(model_info)
#     val_file.write('val_loss: '+str(val_loss))
#     val_file.write('\n\n')
#     val_file.close()



def save_drug_drug_link_probability(pos_df, neg_df,param_dict, out_dir):
    #inputs: df with link prediction probability after applying sigmoid on models predicted score for an edges(positive, negative)
    param_str = dict_to_str(param_dict)

    pos_out_file = out_dir+'pos_val_scores'+ param_str+'.tsv'
    neg_out_file = out_dir +  'neg_val_scores'+param_str+'.tsv'

    os.makedirs(os.path.dirname(pos_out_file), exist_ok=True)
    os.makedirs(os.path.dirname(neg_out_file), exist_ok=True)

    pos_df.to_csv(pos_out_file, sep='\t')
    neg_df.to_csv(neg_out_file, sep='\t')




def save_model_info_with_loss(min_val_loss,param_dict, out_dir, fold_no):
    #inputs: df with link prediction probability after applying sigmoid on models predicted score for an edges(positive, negative)
    param_str = dict_to_str(param_dict)

    out_file = out_dir + param_str + '_model_val_loss.txt'

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    if fold_no==0:
        f = open(out_file, "w")
    else:
        f = open(out_file, "a")
    # f.write('model_info: ' +str(model_info))
    f.write('val_loss: '+str(min_val_loss))
    f.write('\n\n')
    f.close()



def dict_to_str(param_dict):
    param_str = ''
    for key in param_dict:
        param_str += '_' + key + '_' + str(param_dict[key])
    return param_str


def gpu_memory_usage():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    # print("Total memory:", (info.total)/(1024*1024))
    print("Free memory:", (info.free)/(1024*1024))
    print("Used memory:", (info.used)/(1024*1024))

    nvidia_smi.nvmlShutdown()

