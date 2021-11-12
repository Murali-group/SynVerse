import pandas as pd
import numpy as np
import pubchempy as pcp
import time
import os

def drugname_to_pubchemid(drug_name):
    for i in range(0, 5):
        try:
            compounds = pcp.get_compounds(drug_name, 'name')
            if (len(compounds) > 0):
                # return the first compound from the search result
                print(compounds[0].cid)
                return compounds[0].cid
            else:
                print('None')
                return None
        except pcp.PubChemHTTPError:
            print('PubChemHTTPError')
            time.sleep(0.3)
            continue
    return None


drug_target_file = "/home/tasnina/Projects/SynVerse/inputs/drugs/drug_target_map.tsv"

deepsynergy_data_dir = '/home/tasnina/Projects/SynVerse/datasets/deepsynergy/'
ds_synergy_file = deepsynergy_data_dir + 'labels.csv'
drug_name_to_cid_map_file = deepsynergy_data_dir + 'drug_name_to_cid_map.tsv'

ds_synergy_df = pd.read_csv(ds_synergy_file)
ds_cell_lines = set(ds_synergy_df['cell_line'])

if not os.path.exists(drug_name_to_cid_map_file):
    ds_drugs_names = set(list(ds_synergy_df['drug_a_name']) + list(ds_synergy_df['drug_b_name']))

    ds_drugs_name_to_cid = {}
    for drug_name in ds_drugs_names:
        cid = drugname_to_pubchemid(drug_name)
        ds_drugs_name_to_cid[drug_name] = cid

    ds_drugs_name_to_cid_df = pd.DataFrame({'drug_name': ds_drugs_name_to_cid.keys(),
                                            'pubchem_cid': ds_drugs_name_to_cid.values()})
    ds_drugs_name_to_cid_df.to_csv(drug_name_to_cid_map_file, sep='\t')

ds_drugs_name_to_cid_df = pd.read_csv(drug_name_to_cid_map_file, dtype=str, sep='\t', index_col=0)

ds_drugs_name_to_cid_dict = dict(zip(ds_drugs_name_to_cid_df['drug_name'],ds_drugs_name_to_cid_df['pubchem_cid'] ))
ds_synergy_df['drug_a_cid'] = ds_synergy_df['drug_a_name'].apply(lambda x: ds_drugs_name_to_cid_dict[x])
ds_synergy_df['drug_b_cid'] = ds_synergy_df['drug_b_name'].apply(lambda x: ds_drugs_name_to_cid_dict[x])

ds_drug_cids = set(ds_drugs_name_to_cid_df['pubchem_cid'].astype(str))


drug_target_df = pd.read_csv(drug_target_file, sep='\t', index_col=0)
drug_with_target_cids = set(drug_target_df['pubchem_cid'])

common_cids = ds_drug_cids.intersection(drug_with_target_cids)




#now see how many synergistic drug pairs are being covered by this 31 drugs
print('init drug-drug pairs in deepsynergy:', len(ds_synergy_df))
ds_synergy_df = ds_synergy_df[(ds_synergy_df['drug_a_cid'].isin(common_cids)) &\
                              (ds_synergy_df['drug_b_cid'].isin(common_cids))]

print('drug-drug pairs in deepsynergy after taking only common drugs:', len(ds_synergy_df))

print('number of cell lines in deepsynergy after taking only common drugs:', len(set(ds_synergy_df['cell_line'])))
print(ds_synergy_df.head())