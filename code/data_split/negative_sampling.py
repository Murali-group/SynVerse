import numpy as np
import time
import random
import pandas as pd
import copy
import os


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

    print('total drugs in cell_line %s is %d',(cell_line, len(drug_dict.keys())))
    for d in drug_dict.keys():
        # print('drug: ', d)
        m = drug_dict[d]

        # print('drug: ', d, m)
        total_neg_sample = int(m*neg_fact)
        n = 2

        while True:
            #generate pairs having drug_1=d and some random drug as drug_2. generate n times pairs then that is needed.
            idx_i = [d] * total_neg_sample * n
            # idx_j = random.choices(list(drug_degrees_dict.keys()), k=total_neg_sample* n, \
            #                        weights=list(drug_degrees_dict.values()))

            # print(len(drugs), total_neg_sample*n)
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
            if n>20:
               if n%20==0:
                    print(len(drugs), m, total_neg_sample, len(new_false_val_edges))

    val_edges_false = np.array(list(val_edges_false))
    neg_df = pd.DataFrame({'Drug1_pubchem_cid': val_edges_false[:, 0], \
                           'Drug2_pubchem_cid': val_edges_false[:, 1], \
                           'Cell_line': [cell_line] * len(val_edges_false), \
                           'Loewe_label': [0] * len(val_edges_false)})
    # print(len(neg_df), int(len(cell_line_spec_drug_drug_pairs_set)*neg_fact) )
    # assert len(neg_df) == int(len(cell_line_spec_drug_drug_pairs_set)*neg_fact), 'problem negative sampling'
    # assert init_len == len(cell_line_spec_drug_drug_pairs_set), 'problem data prep'

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
            if n>20:
                print('n: ', n, d, m )


    val_edges_false = np.array(list(val_edges_false))
    neg_df = pd.DataFrame({'Drug1_pubchem_cid': val_edges_false[:, 0], \
                           'Drug2_pubchem_cid': val_edges_false[:, 1], \
                           'Cell_line': [cell_line] * number_of_pairs_in_cell_line*neg_fact, \
                           'Loewe_label': [0] * number_of_pairs_in_cell_line*neg_fact})

    assert len(neg_df) == len(cell_line_spec_drug_drug_pairs_set)*neg_fact, 'problem negative sampling'
    assert init_len == len(cell_line_spec_drug_drug_pairs_set), 'problem data prep'

    return neg_df


# def drug_based_negative_sampling_semi_random_alt(folds, synergy_df, neg_fact, number_of_folds):
#     # if (x,y) is in 'cell_line_spec_drug_drug_pairs_set', both (x,y) and (y,x) should not be in val_edges_false.
#     #that's why I used sorting here.
#     start_idx = 0
#     all_neg_df = pd.DataFrame()
#     neg_folds = {i: {'test': [], 'train': [], 'val': []} for i in range(number_of_folds)}
#     for fold in folds:
#         temp_drug_track = {}
#         for split_type in ['train', 'test', 'val']:
#
#             df_1 = synergy_df[synergy_df.index.isin(folds[fold][split_type])]
#             cell_lines = list(df_1['Cell_line'].unique())
#             neg_df = pd.DataFrame()
#             for cell_line in cell_lines:
#                 edges_false = set()
#                 df = df_1[df_1['Cell_line']==cell_line]
#
#
#                 drugs = list(set(df['Drug1_pubchem_cid']).\
#                              union(set(df['Drug2_pubchem_cid'])))
#
#                 temp_drug_track[split_type] = drugs
#
#                 print('total drugs: ', len(drugs), 'total pos pairs: ', len(df))
#
#                 #pos_drug_drug_pairs_set is just for making sure that the pos and neg edges do not overlap
#                 pos_drug_drug_pairs_set = set(zip(df['Drug1_pubchem_cid'], df['Drug2_pubchem_cid']))
#                 pos_drug_drug_pairs_set = set([(max(idx_1, idx_2), min(idx_1, idx_2))\
#                                                for idx_1, idx_2 in pos_drug_drug_pairs_set])
#
#                 total_pos_number_of_pairs = len(pos_drug_drug_pairs_set)
#
#
#
#                 drug_count = df.groupby(['Drug1_pubchem_cid']).size().reset_index(name='counts')
#                 drug_dict = dict(zip(drug_count['Drug1_pubchem_cid'],drug_count['counts']))
#
#                 # print('total drugs in cell_line %s is %d',(cell_line, len(drug_dict.keys())))
#                 for d in drug_dict.keys():
#                     # print('drug: ', d)
#                     m = drug_dict[d]
#
#                     # print('drug: ', d, m)
#                     total_neg_sample = m*neg_fact
#                     n = 2
#                     while True:
#                         #generate pairs having drug_1=d and some random drug as drug_2. generate n times pairs then that is needed.
#                         idx_i = [d] * total_neg_sample * n
#                         # idx_j = random.choices(list(drug_degrees_dict.keys()), k=total_neg_sample* n, \
#                         #                        weights=list(drug_degrees_dict.values()))
#                         idx_j = random.choices(drugs, k=total_neg_sample* n)
#                         # chosen_cell_lines = random.choices(cell_lines, k=total_neg_sample* n)
#                         new_edges = set(zip(idx_i, idx_j))
#
#                         # sort and #remove (x,x) pairs i.e. where two drugs are the same
#                         new_edges = set([(max(idx_1, idx_2), min(idx_1, idx_2)) \
#                                          for idx_1, idx_2 in new_edges if idx_1 != idx_2])
#                         new_false_edges = new_edges.difference(pos_drug_drug_pairs_set)
#                         # print('size of new_false_edges: ', len(new_false_edges))
#                         new_false_edges = new_false_edges.difference(edges_false)
#                         n += 1
#
#                         # print('sampled edges: ', len(new_false_val_edges))
#                         if (len(new_false_edges)>=total_neg_sample):
#                             edges_false = edges_false.union(set(list(new_false_edges)[0:total_neg_sample]))
#
#                             break
#                         if (n>20):
#
#                             s = set([(d1, d2) for d1, d2 in pos_drug_drug_pairs_set if (d1==d) | (d2==d)])
#                             print('n: ', n, d, len(drugs),len(s), m)
#
#
#
#                 edges_false = np.array(list(edges_false))
#
#                 # print('length of false edges: ', len(edges_false))
#                 # print('total_pos_number_of_pairs: ', total_pos_number_of_pairs)
#
#                 cell_line_neg_df = pd.DataFrame({'Drug1_pubchem_cid': edges_false[:, 0],
#                                        'Drug2_pubchem_cid': edges_false[:, 1],
#                                        'Cell_line': [cell_line]* total_pos_number_of_pairs*neg_fact,
#                                        'Loewe_label': [0] * total_pos_number_of_pairs*neg_fact})
#
#                 neg_df = pd.concat([neg_df, cell_line_neg_df], axis=0)
#
#             neg_df.reset_index(drop=True, inplace=True)
#             # assert len(neg_df) == total_pos_number_of_pairs*neg_fact, 'problem negative sampling'
#
#             all_neg_df = pd.concat([all_neg_df, neg_df], axis=0)
#             end_idx = start_idx + total_pos_number_of_pairs*neg_fact
#
#             neg_folds[fold][split_type] = list(range(start_idx, end_idx, 1))
#
#             start_idx = end_idx
#
#         print('common drug train test: ', set(temp_drug_track['train']).intersection(set(temp_drug_track['test'])))
#         print('common drug train test: ', set(temp_drug_track['train']).intersection(set(temp_drug_track['val'])))
#
#         all_neg_df.reset_index(drop=True,inplace=True)
#
#         # print('length of total neg_df: ', list(all_neg_df.index))
#     return neg_folds, all_neg_df


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
