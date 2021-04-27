import numpy as np
import time
import random
import pandas as pd
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


def negative_sampling(cell_line, synergy_df, neg_fact):
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





# def negative_sampling(cell_line, synergy_df):
#     # if (x,y) is in 'cell_line_spec_drug_drug_pairs_set', both (x,y) and (y,x) should not be in val_edges_false.
#     #that's why I used sorting here.
#     drugs = list(set(synergy_df['Drug1_pubchem_cid']). \
#                  union(set(synergy_df['Drug2_pubchem_cid'])))
#
#     cell_line_spec_synergy_df = synergy_df[synergy_df['Cell_line']==cell_line]
#     cell_line_spec_drug_drug_pairs_set = set(zip(cell_line_spec_synergy_df['Drug1_pubchem_cid'],\
#                                                  cell_line_spec_synergy_df['Drug2_pubchem_cid']))
#
#     init_len = len(cell_line_spec_drug_drug_pairs_set) #this is just for sanity check
#     cell_line_spec_drug_drug_pairs_set = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in \
#                                               cell_line_spec_drug_drug_pairs_set])
#
#     number_of_pairs_in_cell_line = len(cell_line_spec_drug_drug_pairs_set)
#     drug_degrees_dict = find_drug_degrees(drugs, cell_line_spec_drug_drug_pairs_set)
#
#     val_edges_false = set()
#     count=0
#
#     # idx_i = list(cell_line_spec_synergy_df['Drug1_pubchem_cid'])
#     while True:
#         # idx_i = random.choices(synergy_df['Drug1_pubchem_cid'].unique(), k=int(number_of_pairs_in_cell_line))
#         # idx_j = random.choices(synergy_df['Drug2_pubchem_cid'].unique(), k=int(number_of_pairs_in_cell_line))
#         # idx_i = random.choices(list(drug_degrees_dict.keys()), k=number_of_pairs_in_cell_line,\
#         #                        weights=list(drug_degrees_dict.values()))

#


#
#         idx_i = random.choices(list(drug_degrees_dict.keys()), k=number_of_pairs_in_cell_line, \
#                                 weights=list(drug_degrees_dict.values()))
#         # idx_i = list(cell_line_spec_synergy_df['Drug1_pubchem_cid'])
#         idx_j = random.choices(list(drug_degrees_dict.keys()), k=number_of_pairs_in_cell_line,\
#                                weights=list(drug_degrees_dict.values()))
#
#         new_val_edges = set(zip(idx_i, idx_j))
#
#         #sort and #remove (x,x) pairs i.e. where two drugs are the same
#         new_val_edges = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in new_val_edges  if idx_1!=idx_2])
#
#         new_false_val_edges = new_val_edges.difference(cell_line_spec_drug_drug_pairs_set)
#
#         # idx_1 = [drug1 for drug1,drug2 in new_val_edges.difference(new_false_val_edges)]
#
#         val_edges_false = val_edges_false.union(new_false_val_edges)
#
#         if len(val_edges_false) >= int(number_of_pairs_in_cell_line):
#             # print('Val false edges done')
#             val_edges_false = np.array(list(val_edges_false)[0:int(number_of_pairs_in_cell_line)])
#             neg_df = pd.DataFrame({'Drug1_pubchem_cid': val_edges_false[:, 0], \
#                                    'Drug2_pubchem_cid': val_edges_false[:, 1], \
#                                    'Cell_line': [cell_line] * number_of_pairs_in_cell_line, \
#                                    'Loewe_label': [0] * number_of_pairs_in_cell_line})
#             break
#         # print('count: ', count, 'sampled neg: ', len(val_edges_false))
#         print('count: ', count)
#
#         count+=1
#     assert len(neg_df) == len(cell_line_spec_drug_drug_pairs_set), 'problem negative sampling'
#     assert init_len == len(cell_line_spec_drug_drug_pairs_set), 'problem data prep'
#
#     return neg_df
#
