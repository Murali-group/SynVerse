from preprocessing.drug_preprocess import prepare_drug_features
from preprocessing.cell_line_preprocess import prepare_cell_line_features
from preprocessing.synergy_data_preprocess import *
from preprocessing.autoencoder import autoencoder_runner
from preprocessing.preprocess_utils import *

from utils import *
from split import *
import pandas as pd

def load_filter_triplets_features(synergy_df, drug_pids, cell_line_names,inputs, params,device, feat_filt=True, abundance_based_filt=True):

    # ********************************** GET FEATURES READY *******************************************************
    ''' Read parsed drug features and do user-chosen filtering and preprocessing.'''
    dfeat_dict, dfeat_names = prepare_drug_features(drug_pids, params, inputs, device)

    ''' Read parsed cell line features and do user-chosen filtering and preprocessing.'''
    cfeat_dict, cfeat_names = prepare_cell_line_features(cell_line_names, params, inputs, device)

    if feat_filt:
        '''Filter out the triplets based on the availability of drug and cell line features'''
        synergy_df = feature_based_filtering(synergy_df, dfeat_dict['value'], cfeat_dict['value'])

    if abundance_based_filt:
        '''keep the cell lines consisting of at least abundance% of the total #triplets in the final dataset.'''
        synergy_df = abundance_based_filtering(synergy_df, min_frac=params.abundance)

    '''get drug_pid to index and cell_line_name to index mapping'''
    feat_str = get_feat_prefix(dfeat_dict, cfeat_dict)
    dir_for_saving_indx_map = f'{params.split_dir}/{feat_str}/k_{params.abundance}_{params.score_name}/'
    synergy_df, drug_2_idx, cell_line_2_idx = generalize_data(synergy_df, dir_for_saving_indx_map)
    print_synergy_stat(synergy_df)

    ''' Sort the features of drugs and cell lines according to their index'''
    dfeat_dict['value'], cfeat_dict['value'] = get_index_sorted_feature_matrix(dfeat_dict['value'], drug_2_idx,
                                                                               cfeat_dict['value'], cell_line_2_idx)

    return synergy_df, dfeat_dict, cfeat_dict, drug_2_idx, cell_line_2_idx


def post_split_processing(dfeat_dict, cfeat_dict, all_train_df, params, split_info_str,device):
    '''
    Any preprocessing of features that needs to be done differently on train and test drugs and cell lines should be done inside this function. These preprocessing has to be done after data splitting.
    For example, for standardizing drug features we need to find the mean and std on training drugs and then use that mean and std to standardize both train and test drugs.
    :return:
    '''

    # convert feature dataframes into numpy arrays while in the array row i corresponds to the drug with numerical idx i
    cur_dfeat_dict = copy.deepcopy(dfeat_dict)
    cur_cfeat_dict = copy.deepcopy(cfeat_dict)

    # Reduce dimension of data or compress data using autoenencoder
    train_drug_idx = list(set(all_train_df['source']).union(set(all_train_df['target'])))
    train_cell_idx = list(set(all_train_df['edge_type']).union(set(all_train_df['edge_type'])))

    cur_dfeat_dict['value'], cur_dfeat_dict['dim'] = autoencoder_runner(cur_dfeat_dict['value'], cur_dfeat_dict['dim'],
                                                                        cur_dfeat_dict['compress'],
                                                                        train_drug_idx,
                                                                        hidden_dim_options=params.autoencoder_dims,
                                                                        epoch=500,
                                                                        file_prefix=f'{params.input_dir}/drug/AE/{split_info_str}/',
                                                                        device=device, force_run=False)
    cur_cfeat_dict['value'], cur_cfeat_dict['dim'] = autoencoder_runner(cur_cfeat_dict['value'], cur_cfeat_dict['dim'],
                                                                        cur_cfeat_dict['compress'],
                                                                        train_cell_idx,
                                                                        hidden_dim_options=params.autoencoder_dims,
                                                                        epoch=500,
                                                                        file_prefix=f'{params.input_dir}/cell-line/AE/{split_info_str}/',
                                                                        device=device, force_run=False)

    # Normalize data based on training data. Use the computed mean, std from training data to normalize test data.
    cur_dfeat_dict['value'], cur_cfeat_dict['value'] = normalization_wrapper(cur_dfeat_dict['value'],
                                                                             cur_cfeat_dict['value'],
                                                                             cur_dfeat_dict['norm'],
                                                                             cur_cfeat_dict['norm'],
                                                                             all_train_df)

    return cur_dfeat_dict, cur_cfeat_dict