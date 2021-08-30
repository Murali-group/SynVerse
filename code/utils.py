import numpy as np
import time
import random
import pandas as pd
import copy
import os
# import torch
# import torch.nn as nn
# dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader
# import pickle

def find_drug_degrees(drugs, cell_line_spec_drug_drug_pairs_set):
    #return a dictionary: where, key = drugs in the cell line specific synergy_df; value: degree of key drug in the cell line\
    #under consideration

    drug_degrees = {drug: 0.1 for drug in drugs}

    # t1 = time.time()
    # print('total pairs: ', len(cell_line_spec_drug_drug_pairs_set), 'total drugs: ', len(drugs))
    # print(t1)
    for drug1, drug2 in cell_line_spec_drug_drug_pairs_set:
        drug_degrees[drug1] += 1
        drug_degrees[drug2] += 1

    return drug_degrees


def cell_line_based_negative_sampling_semi_random(cell_line, synergy_df, neg_fact):
    # if (x,y) is in 'cell_line_spec_drug_drug_pairs_set', both (x,y) and (y,x) should not be in val_edges_false.
    #that's why I used sorting here.
    drugs = list(set(synergy_df['Drug1_pubchem_cid']).\
                 union(set(synergy_df['Drug2_pubchem_cid'])))
    print('total drugs: ', len(drugs))
    cell_line_spec_synergy_df = synergy_df[synergy_df['Cell_line']==cell_line]
    cell_line_spec_drug_drug_pairs_set = set(zip(cell_line_spec_synergy_df['Drug1_pubchem_cid'],\
                                                 cell_line_spec_synergy_df['Drug2_pubchem_cid']))

    init_len = len(cell_line_spec_drug_drug_pairs_set) #this is just for sanity check
    cell_line_spec_drug_drug_pairs_set = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in \
                                              cell_line_spec_drug_drug_pairs_set])

    number_of_pairs_in_cell_line = len(cell_line_spec_drug_drug_pairs_set)
    # drug_degrees_dict = find_drug_degrees(drugs, cell_line_spec_drug_drug_pairs_set)

    val_edges_false = set()


    drug_count_per_cell_line = cell_line_spec_synergy_df.groupby(['Drug1_pubchem_cid']).size().reset_index(name='counts')
    drug_dict = dict(zip(drug_count_per_cell_line['Drug1_pubchem_cid'],drug_count_per_cell_line['counts']))

    # print('total drugs in cell_line %s is %d',(cell_line, len(drug_dict.keys())))
    for d in drug_dict.keys():
        # print('drug: ', d)
        m = drug_dict[d]

        # print('drug: ', d, m)
        total_neg_sample = m*neg_fact
        n = 2
        while True:
            #generate pairs having drug_1=d and some random drug as drug_2. generate n times pairs then that is needed.
            idx_i = [d] * total_neg_sample * n
            # idx_j = random.choices(list(drug_degrees_dict.keys()), k=total_neg_sample* n, \
            #                        weights=list(drug_degrees_dict.values()))
            idx_j = random.choices(drugs, k=total_neg_sample* n)
            new_val_edges = set(zip(idx_i, idx_j))

            # sort and #remove (x,x) pairs i.e. where two drugs are the same
            new_val_edges = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in new_val_edges if idx_1 != idx_2])
            new_false_val_edges = new_val_edges.difference(cell_line_spec_drug_drug_pairs_set)
            new_false_val_edges = new_false_val_edges.difference(val_edges_false)
            n += 1

            # print('sampled edges: ', len(new_false_val_edges))
            if (len(new_false_val_edges)>=total_neg_sample):
                val_edges_false = val_edges_false.union(set(list(new_false_val_edges)[0:total_neg_sample]))

                break
            # if n>20:
                # print('n: ', n, d, m )


    val_edges_false = np.array(list(val_edges_false))
    neg_df = pd.DataFrame({'Drug1_pubchem_cid': val_edges_false[:, 0], \
                           'Drug2_pubchem_cid': val_edges_false[:, 1], \
                           'Cell_line': [cell_line] * number_of_pairs_in_cell_line*neg_fact, \
                           'Loewe_label': [0] * number_of_pairs_in_cell_line*neg_fact})

    assert len(neg_df) == len(cell_line_spec_drug_drug_pairs_set)*neg_fact, 'problem negative sampling'
    assert init_len == len(cell_line_spec_drug_drug_pairs_set), 'problem data prep'

    return neg_df



def cell_line_based_negative_sampling_degree_based(cell_line, synergy_df, neg_fact):
    # if (x,y) is in 'cell_line_spec_drug_drug_pairs_set', both (x,y) and (y,x) should not be in val_edges_false.
    #that's why I used sorting here.
    drugs = list(set(synergy_df['Drug1_pubchem_cid']).\
                 union(set(synergy_df['Drug2_pubchem_cid'])))

    cell_line_spec_synergy_df = synergy_df[synergy_df['Cell_line']==cell_line]
    cell_line_spec_drug_drug_pairs_set = set(zip(cell_line_spec_synergy_df['Drug1_pubchem_cid'],\
                                                 cell_line_spec_synergy_df['Drug2_pubchem_cid']))

    init_len = len(cell_line_spec_drug_drug_pairs_set) #this is just for sanity check
    cell_line_spec_drug_drug_pairs_set = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in \
                                              cell_line_spec_drug_drug_pairs_set])

    number_of_pairs_in_cell_line = len(cell_line_spec_drug_drug_pairs_set)
    drug_degrees_dict = find_drug_degrees(drugs, cell_line_spec_drug_drug_pairs_set)

    val_edges_false = set()


    drug_count_per_cell_line = cell_line_spec_synergy_df.groupby(['Drug1_pubchem_cid']).size().reset_index(name='counts')
    drug_dict = dict(zip(drug_count_per_cell_line['Drug1_pubchem_cid'],drug_count_per_cell_line['counts']))

    # print('total drugs in cell_line %s is %d',(cell_line, len(drug_dict.keys())))
    for d in drug_dict.keys():
        m = drug_dict[d]
        total_neg_sample = m*neg_fact
        n = 2
        while True:
            #generate pairs having drug_1=d and some random drug as drug_2. generate n times pairs then that is needed.
            idx_i = [d] * total_neg_sample * n
            idx_j = random.choices(list(drug_degrees_dict.keys()), k=total_neg_sample* n, \
                                   weights=list(drug_degrees_dict.values()))
            new_val_edges = set(zip(idx_i, idx_j))

            # sort and #remove (x,x) pairs i.e. where two drugs are the same
            new_val_edges = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in new_val_edges if idx_1 != idx_2])
            new_false_val_edges = new_val_edges.difference(cell_line_spec_drug_drug_pairs_set)
            new_false_val_edges = new_false_val_edges.difference(val_edges_false)
            n += 1
            if (len(new_false_val_edges)>=total_neg_sample):
                val_edges_false = val_edges_false.union(set(list(new_false_val_edges)[0:total_neg_sample]))
                break
            # if n>20:
                # print('n: ', n, d, m )


    val_edges_false = np.array(list(val_edges_false))
    neg_df = pd.DataFrame({'Drug1_pubchem_cid': val_edges_false[:, 0], \
                           'Drug2_pubchem_cid': val_edges_false[:, 1], \
                           'Cell_line': [cell_line] * number_of_pairs_in_cell_line*neg_fact, \
                           'Loewe_label': [0] * number_of_pairs_in_cell_line*neg_fact})

    assert len(neg_df) == len(cell_line_spec_drug_drug_pairs_set)*neg_fact, 'problem negative sampling'
    assert init_len == len(cell_line_spec_drug_drug_pairs_set), 'problem data prep'

    return neg_df


def drug_based_negative_sampling_semi_random_alt(folds, synergy_df, neg_fact, number_of_folds):
    # if (x,y) is in 'cell_line_spec_drug_drug_pairs_set', both (x,y) and (y,x) should not be in val_edges_false.
    #that's why I used sorting here.
    start_idx = 0
    all_neg_df = pd.DataFrame()
    neg_folds = {i: {'test': [], 'train': [], 'val': []} for i in range(number_of_folds)}
    for fold in folds:
        temp_drug_track = {}
        for split_type in ['train', 'test', 'val']:

            df_1 = synergy_df[synergy_df.index.isin(folds[fold][split_type])]
            cell_lines = list(df_1['Cell_line'].unique())
            neg_df = pd.DataFrame()
            for cell_line in cell_lines:
                edges_false = set()
                df = df_1[df_1['Cell_line']==cell_line]


                drugs = list(set(df['Drug1_pubchem_cid']).\
                             union(set(df['Drug2_pubchem_cid'])))

                temp_drug_track[split_type] = drugs

                print('total drugs: ', len(drugs), 'total pos pairs: ', len(df))

                #pos_drug_drug_pairs_set is just for making sure that the pos and neg edges do not overlap
                pos_drug_drug_pairs_set = set(zip(df['Drug1_pubchem_cid'], df['Drug2_pubchem_cid']))
                pos_drug_drug_pairs_set = set([(max(idx_1, idx_2), min(idx_1, idx_2))\
                                               for idx_1, idx_2 in pos_drug_drug_pairs_set])

                total_pos_number_of_pairs = len(pos_drug_drug_pairs_set)



                drug_count = df.groupby(['Drug1_pubchem_cid']).size().reset_index(name='counts')
                drug_dict = dict(zip(drug_count['Drug1_pubchem_cid'],drug_count['counts']))

                # print('total drugs in cell_line %s is %d',(cell_line, len(drug_dict.keys())))
                for d in drug_dict.keys():
                    # print('drug: ', d)
                    m = drug_dict[d]

                    # print('drug: ', d, m)
                    total_neg_sample = m*neg_fact
                    n = 2
                    while True:
                        #generate pairs having drug_1=d and some random drug as drug_2. generate n times pairs then that is needed.
                        idx_i = [d] * total_neg_sample * n
                        # idx_j = random.choices(list(drug_degrees_dict.keys()), k=total_neg_sample* n, \
                        #                        weights=list(drug_degrees_dict.values()))
                        idx_j = random.choices(drugs, k=total_neg_sample* n)
                        # chosen_cell_lines = random.choices(cell_lines, k=total_neg_sample* n)
                        new_edges = set(zip(idx_i, idx_j))

                        # sort and #remove (x,x) pairs i.e. where two drugs are the same
                        new_edges = set([(max(idx_1, idx_2), min(idx_1, idx_2)) \
                                         for idx_1, idx_2 in new_edges if idx_1 != idx_2])
                        new_false_edges = new_edges.difference(pos_drug_drug_pairs_set)
                        # print('size of new_false_edges: ', len(new_false_edges))
                        new_false_edges = new_false_edges.difference(edges_false)
                        n += 1

                        # print('sampled edges: ', len(new_false_val_edges))
                        if (len(new_false_edges)>=total_neg_sample):
                            edges_false = edges_false.union(set(list(new_false_edges)[0:total_neg_sample]))

                            break
                        if (n>20):

                            s = set([(d1, d2) for d1, d2 in pos_drug_drug_pairs_set if (d1==d) | (d2==d)])
                            print('n: ', n, d, len(drugs),len(s), m)



                edges_false = np.array(list(edges_false))

                # print('length of false edges: ', len(edges_false))
                # print('total_pos_number_of_pairs: ', total_pos_number_of_pairs)

                cell_line_neg_df = pd.DataFrame({'Drug1_pubchem_cid': edges_false[:, 0],
                                       'Drug2_pubchem_cid': edges_false[:, 1],
                                       'Cell_line': [cell_line]* total_pos_number_of_pairs*neg_fact,
                                       'Loewe_label': [0] * total_pos_number_of_pairs*neg_fact})

                neg_df = pd.concat([neg_df, cell_line_neg_df], axis=0)

            neg_df.reset_index(drop=True, inplace=True)
            # assert len(neg_df) == total_pos_number_of_pairs*neg_fact, 'problem negative sampling'

            all_neg_df = pd.concat([all_neg_df, neg_df], axis=0)
            end_idx = start_idx + total_pos_number_of_pairs*neg_fact

            neg_folds[fold][split_type] = list(range(start_idx, end_idx, 1))

            start_idx = end_idx

        print('common drug train test: ', set(temp_drug_track['train']).intersection(set(temp_drug_track['test'])))
        print('common drug train test: ', set(temp_drug_track['train']).intersection(set(temp_drug_track['val'])))

        all_neg_df.reset_index(drop=True,inplace=True)

        # print('length of total neg_df: ', list(all_neg_df.index))
    return neg_folds, all_neg_df


def drug_based_negative_sampling_semi_random(folds, synergy_df, neg_fact, number_of_folds):
    # if (x,y) is in 'cell_line_spec_drug_drug_pairs_set', both (x,y) and (y,x) should not be in val_edges_false.
    #that's why I used sorting here.
    start_idx = 0
    all_neg_df = pd.DataFrame()
    neg_folds = {i: {'test': [], 'train': [], 'val': []} for i in range(number_of_folds)}
    for fold in folds:
        temp_drug_track = {}
        for split_type in ['train', 'test', 'val']:
            df = synergy_df[synergy_df.index.isin(folds[fold][split_type])]
            drugs = list(set(df['Drug1_pubchem_cid']).\
                         union(set(df['Drug2_pubchem_cid'])))

            temp_drug_track[split_type] = drugs
            cell_lines = list(df['Cell_line'].unique())
            print('total drugs: ', len(drugs))

            #pos_drug_drug_pairs_set is just for making sure that the pos and neg edges do not overlap
            pos_drug_drug_pairs_set = set(zip(df['Drug1_pubchem_cid'], df['Drug2_pubchem_cid'], df['Cell_line']))
            pos_drug_drug_pairs_set = set([(max(idx_1, idx_2), min(idx_1, idx_2), cell_line)\
                                           for idx_1, idx_2, cell_line in pos_drug_drug_pairs_set])

            total_pos_number_of_pairs = len(pos_drug_drug_pairs_set)
            edges_false = set()


            drug_count = df.groupby(['Drug1_pubchem_cid']).size().reset_index(name='counts')
            drug_dict = dict(zip(drug_count['Drug1_pubchem_cid'],drug_count['counts']))

            # print('total drugs in cell_line %s is %d',(cell_line, len(drug_dict.keys())))
            for d in drug_dict.keys():
                # print('drug: ', d)
                m = drug_dict[d]

                # print('drug: ', d, m)
                total_neg_sample = m*neg_fact
                n = 2
                while True:
                    #generate pairs having drug_1=d and some random drug as drug_2. generate n times pairs then that is needed.
                    idx_i = [d] * total_neg_sample * n
                    # idx_j = random.choices(list(drug_degrees_dict.keys()), k=total_neg_sample* n, \
                    #                        weights=list(drug_degrees_dict.values()))
                    idx_j = random.choices(drugs, k=total_neg_sample* n)
                    chosen_cell_lines = random.choices(cell_lines, k=total_neg_sample* n)
                    new_edges = set(zip(idx_i, idx_j, chosen_cell_lines))

                    # sort and #remove (x,x) pairs i.e. where two drugs are the same
                    new_edges = set([(max(idx_1, idx_2), min(idx_1, idx_2), cell_line) \
                                     for idx_1, idx_2, cell_line in new_edges if idx_1 != idx_2])
                    new_false_edges = new_edges.difference(pos_drug_drug_pairs_set)
                    new_false_edges = new_false_edges.difference(edges_false)
                    n += 1

                    # print('sampled edges: ', len(new_false_val_edges))
                    if (len(new_false_edges)>=total_neg_sample):
                        edges_false = edges_false.union(set(list(new_false_edges)[0:total_neg_sample]))

                        break
                    # if n>20:
                        # print('n: ', n, d, m )


            edges_false = np.array(list(edges_false))

            print('length of false edges: ', len(edges_false))
            print('total_pos_number_of_pairs: ', total_pos_number_of_pairs)

            neg_df = pd.DataFrame({'Drug1_pubchem_cid': edges_false[:, 0],
                                   'Drug2_pubchem_cid': edges_false[:, 1],
                                   'Cell_line': edges_false[:, 2],
                                   'Loewe_label': [0] * total_pos_number_of_pairs*neg_fact})

            assert len(neg_df) == total_pos_number_of_pairs*neg_fact, 'problem negative sampling'

            all_neg_df = pd.concat([all_neg_df, neg_df], axis=0)
            end_idx = start_idx + total_pos_number_of_pairs*neg_fact

            neg_folds[fold][split_type] = list(range(start_idx, end_idx, 1))

            start_idx = end_idx

        print('common drug train test: ', set(temp_drug_track['train']).intersection(set(temp_drug_track['test'])))
        print('common drug train test: ', set(temp_drug_track['train']).intersection(set(temp_drug_track['val'])))

        all_neg_df.reset_index(drop=True,inplace=True)

        # print('length of total neg_df: ', list(all_neg_df.index))
    return neg_folds, all_neg_df


def drug_based_negative_sampling_degree_based(folds, synergy_df, neg_fact, number_of_folds):
    # if (x,y) is in 'cell_line_spec_drug_drug_pairs_set', both (x,y) and (y,x) should not be in val_edges_false.
    #that's why I used sorting here.
    start_idx = 0
    all_neg_df = pd.DataFrame()
    neg_folds = {i: {'test': [], 'train': [], 'val': []} for i in range(number_of_folds)}
    for fold in folds:
        for split_type in ['train', 'test', 'val']:
            df = synergy_df[synergy_df.index.isin(folds[fold][split_type])]
            drugs = list(set(df['Drug1_pubchem_cid']).\
                         union(set(df['Drug2_pubchem_cid'])))
            cell_lines = list(df['Cell_line'].unique())
            print('total drugs: ', len(drugs))

            #pos_drug_drug_pairs_set is just for making sure that the pos and neg edges do not overlap
            pos_drug_drug_pairs_set = set(zip(df['Drug1_pubchem_cid'], df['Drug2_pubchem_cid'], df['Cell_line']))
            pos_drug_drug_pairs_set = set([(max(idx_1, idx_2), min(idx_1, idx_2), cell_line)\
                                           for idx_1, idx_2, cell_line in pos_drug_drug_pairs_set])

            total_pos_number_of_pairs = len(pos_drug_drug_pairs_set)
            edges_false = set()

            drug_degrees_dict = find_drug_degrees(drugs, pos_drug_drug_pairs_set)

            drug_count = df.groupby(['Drug1_pubchem_cid']).size().reset_index(name='counts')
            drug_dict = dict(zip(drug_count['Drug1_pubchem_cid'],drug_count['counts']))

            # print('total drugs in cell_line %s is %d',(cell_line, len(drug_dict.keys())))
            for d in drug_dict.keys():
                # print('drug: ', d)
                m = drug_dict[d]

                # print('drug: ', d, m)
                total_neg_sample = m*neg_fact
                n = 2
                while True:
                    #generate pairs having drug_1=d and some random drug as drug_2. generate n times pairs then that is needed.
                    idx_i = [d] * total_neg_sample * n
                    idx_j = random.choices(list(drug_degrees_dict.keys()), k=total_neg_sample* n, \
                                           weights=list(drug_degrees_dict.values()))

                    chosen_cell_lines = random.choices(cell_lines, k=total_neg_sample* n)
                    new_edges = set(zip(idx_i, idx_j, chosen_cell_lines))

                    # sort and #remove (x,x) pairs i.e. where two drugs are the same
                    new_edges = set([(max(idx_1, idx_2), min(idx_1, idx_2), cell_line) \
                                     for idx_1, idx_2, cell_line in new_edges if idx_1 != idx_2])
                    new_false_edges = new_edges.difference(pos_drug_drug_pairs_set)
                    new_false_edges = new_false_edges.difference(edges_false)
                    n += 1

                    # print('sampled edges: ', len(new_false_val_edges))
                    if (len(new_false_edges)>=total_neg_sample):
                        edges_false = edges_false.union(set(list(new_false_edges)[0:total_neg_sample]))

                        break
                    # if n>20:
                        # print('n: ', n, d, m )


            edges_false = np.array(list(edges_false))
            neg_df = pd.DataFrame({'Drug1_pubchem_cid': edges_false[:, 0],
                                   'Drug2_pubchem_cid': edges_false[:, 1],
                                   'Cell_line': edges_false[:, 2],
                                   'Loewe_label': [0] * total_pos_number_of_pairs*neg_fact})

            assert len(neg_df) == len(total_pos_number_of_pairs)*neg_fact, 'problem negative sampling'

            all_neg_df = pd.concat([all_neg_df, neg_df], axis=0)
            end_idx = start_idx + total_neg_sample
            neg_folds[fold][split_type] = range(start_idx, end_idx, 1)
            start_idx = end_idx

    return neg_folds, all_neg_df

def generate_cell_line_idx_mapping(synergy_df):

    cell_lines = synergy_df['Cell_line'].unique()
    cell_line_2_idx = {cell_line: i for i, cell_line in enumerate(cell_lines)}
    idx_2_cell_line = {i: cell_line for i, cell_line in enumerate(cell_lines)}
    return cell_line_2_idx, idx_2_cell_line

def generate_cosmicid_to_idx_mapping(synergy_df):
    #write the code here
    cosmicid_to_idx = {}
    return cosmicid_to_idx


def prepare_synverse_param_settings(config_map, dd_decoder_type):
    synverse_settings = config_map['ml_models_settings']['algs']['synverse']

    h_sizes = synverse_settings['h_sizes'] # only hidden and output_layer
    nndecoder_h_sizes = synverse_settings['nndecoder_h_sizes']
    learning_rates = synverse_settings['learning_rates']
    dropouts = synverse_settings['dropouts']
    use_drug_feat_options = synverse_settings['use_drug_feat_options']
    # gene_exs = synverse_settings['gene_exs']

    epoch = synverse_settings['epochs']
    batch_size = synverse_settings['batch_size']
    bias = synverse_settings['bias']
    patience = synverse_settings['patience']

    synverse_params_list = []

    for use_drug_feat_option in use_drug_feat_options:
        # for gene_ex in gene_exs:
        for h_size in h_sizes:
            for learning_rate in learning_rates:
                for dropout in dropouts:
                    synverse_params= {}
                    synverse_params['h_size'] = h_size
                    synverse_params['learning_rate']= learning_rate
                    synverse_params['dropout'] = dropout
                    synverse_params['use_drug_feat'] = use_drug_feat_option
                    # synverse_params['gene_ex'] = gene_ex
                    synverse_params['epoch'] = epoch
                    synverse_params['batch_size'] = batch_size
                    synverse_params['bias'] = bias
                    synverse_params['patience'] = patience

                    if dd_decoder_type=='nndecoder':
                        for i in range(len(nndecoder_h_sizes)):
                            synverse_params_temp = copy.deepcopy(synverse_params)
                            synverse_params_temp['nndecoder_hsize'] = nndecoder_h_sizes[i]

                            synverse_params_list.append(synverse_params_temp)
                    else:
                        synverse_params_list.append(synverse_params)

    return synverse_params_list

def prepare_deepsynergy_param_settings(config_map):
    ds_settings = config_map['ml_models_settings']['algs']['deepsynergy']

    layers_options = ds_settings['layers']
    learning_rates = ds_settings['learning_rate']

    in_hid_dropouts = ds_settings['in_hid_dropouts']
    #            dropout: [0,0.5]
    batch_size = ds_settings['batch_size']
    epochs=ds_settings['epochs']
    act_func = ds_settings['act_func']
    norm = ds_settings['norm']
    patience = ds_settings['patience']

    ds_params_list = []

    for layers in layers_options:
        for in_hid_dropout in in_hid_dropouts:
            for learning_rate in learning_rates:
                for in_hid_dropout in in_hid_dropouts:
                    ds_params = {}
                    ds_params['layers'] = layers
                    ds_params['learning_rate'] = learning_rate
                    ds_params['in_hid_dropout'] = in_hid_dropout
                    ds_params['act_func'] = act_func
                    ds_params['norm'] = norm
                    ds_params['epochs'] = epochs
                    ds_params['batch_size'] = batch_size
                    ds_params['patience'] = patience

                    ds_params_list.append(ds_params)

    return ds_params_list


def prepare_svm_param_settings(config_map):
    svm_settings = config_map['ml_models_settings']['algs']['svm']
    gamma_list = svm_settings['gamma']
    c_list = svm_settings['c']
    kernel_list = svm_settings['kernel']

    svm_params_list = []
    for gamma in gamma_list:
        for c in c_list:
            for kernel in kernel_list:
                        svm_params = {}
                        svm_params['gamma'] = gamma
                        svm_params['c'] = c
                        svm_params['kernel'] = kernel
                        svm_params_list.append(svm_params)

    return svm_params_list



def prepare_gbr_param_settings(config_map):
    gbr_settings = config_map['ml_models_settings']['algs']['gbr']
    n_estimators_list = gbr_settings['n_estimators']
    learning_rate_list = gbr_settings['learning_rate']
    max_depth_list = gbr_settings['max_depth']


    gbr_params_list = []
    for n_estimators in n_estimators_list:
        for learning_rate in learning_rate_list:
            for max_depth in max_depth_list:

                        gbr_params = {}
                        gbr_params['learning_rate'] = learning_rate
                        gbr_params['n_estimators'] = n_estimators
                        gbr_params['max_depth'] = max_depth
                        gbr_params_list.append(gbr_params)

    return gbr_params_list



