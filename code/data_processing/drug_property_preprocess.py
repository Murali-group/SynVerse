#!/usr/bin/env python
import pubchempy as pcp
import time
import os
import pickle
import pandas as pd
import numpy as np
import deepchem as dcm
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
import requests

def wrapper_download_pubchem_compound(drug_names, drug_name_to_pcomp_file, force_run=True):
    if (os.path.exists(drug_name_to_pcomp_file) and force_run==False):
        with open(drug_name_to_pcomp_file, 'rb') as file:
            drug_name_to_pcomp = pickle.load(file)
        return drug_name_to_pcomp

    else: #either the file is absent or force_run=True. In this case, read from the exixting file and
            #find the remaining pubchem from pubchem website.
        if (os.path.exists(drug_name_to_pcomp_file)):
            with open(drug_name_to_pcomp_file, 'rb') as file:
                drug_name_to_pcomp = pickle.load(file)
        else:
            drug_name_to_pcomp = {}

        #from drug_names, remove the drugs for which I already exytracted pubchem compounds
        drug_names = set(drug_names).difference(set(drug_name_to_pcomp.keys()))

        #Extract puchem compounds from pubchem py
        drug_name_to_pcomp_new, pid_exception, pid_not_found = get_pubchem_compound(drug_names)

        #merge old and newly found compunds into old dict
        drug_name_to_pcomp.update(drug_name_to_pcomp_new)

        # save to pickle file:
        os.makedirs(os.path.dirname(drug_name_to_pcomp_file), exist_ok=True)
        # Open the file in binary write mode
        with open(drug_name_to_pcomp_file, 'wb') as file:
            # Use pickle to dump the dictionary into the file
            pickle.dump(drug_name_to_pcomp, file)

        print('exception: ', str(pid_exception))
        print('Not found: ', str(pid_not_found))

        return drug_name_to_pcomp


def get_pubchem_compound(drug_names, max_retries=15, retry_delay=0.5):
    pubchem_compounds = {}
    exceptions_occurred = []
    cids_not_found = []
    mult_cids = []
    count_processed = 0
    for drug_name in drug_names:
        retries = 0

        while retries < max_retries:
            try:
                # Search for the drug name in PubChem
                results = pcp.get_compounds(drug_name, 'name')
                time.sleep(retry_delay)  # Pause before retrying
                if results:
                    # Get the first result and extract its PubChem ID
                    pubchem_compound = results
                    if len(pubchem_compound)>1: #one drugname to multiple compounds
                        cids = []
                        for i in range (len(pubchem_compound)):
                            cids.append(pubchem_compound[i].cid)
                        print(drug_name, cids)
                        #compare if all the cids from multiple compounds are same or different
                        if (not all(s == cids[0] for s in cids)):
                            mult_cids.append(drug_name)
                    pubchem_compounds[drug_name] = pubchem_compound
                else:
                    cids_not_found.append(drug_name)
                    print('not found: ', drug_name)
                break  # Break the retry loop if successful
            except Exception as e:
                retries += 1
                if retries == max_retries:
                    exceptions_occurred.append(drug_name)
                    print('exception: ', drug_name)
                    break  # Break the retry loop if maximum retries reached
        count_processed += 1
        # print(count_processed)
    return pubchem_compounds, exceptions_occurred, cids_not_found


def find_drugs_with_ueq_smiles(drug_name_to_pcomp):
    '''
    Find the drugs having multiple unequal smiles associated with them.
    :return: List of such drug names.
    '''

    mult_uneq_smiles = []
    mult_smiles = []
    drug_name_2_smiles = {drug_name: [comp.canonical_smiles for comp in drug_name_to_pcomp[drug_name]] for drug_name in
                          drug_name_to_pcomp}

    # check if one drug has multiple smiles
    for drug_name in drug_name_2_smiles.keys():
        if len(drug_name_2_smiles[drug_name]) > 1:
            mult_smiles.append(drug_name)
            if (not all(s == drug_name_2_smiles[drug_name][0] for s in drug_name_2_smiles[drug_name])):
                mult_uneq_smiles.append(drug_name)
                # print(drug_name)
    # assert len(mult_uneq_smiles)==0, print('Some drugs has multiple smiles.', mult_uneq_smiles)
    print('Drugs with multiple different smiles: ', len(mult_uneq_smiles))

    return mult_uneq_smiles


def extract_unambiguouse_drug_pid(drug_name_to_pcomp_file):
    '''
    If a drug name is associated with multiple pids, first we find out if all these pids lead to one/same SMILES.
    If so, we take the first pid and SMILES. If not, we do not map this drug name to any PID at all because of inconsistency.

    '''
    with open(drug_name_to_pcomp_file, 'rb') as file:
        drug_name_to_pcomp = pickle.load(file)
        mult_uneq_smiles = find_drugs_with_ueq_smiles(drug_name_to_pcomp)
        # Now take the first pid and first smiles for each drug
        drug_name_2_pid = {drug_name: str(drug_name_to_pcomp[drug_name][0].cid) for drug_name in
                           drug_name_to_pcomp if drug_name not in mult_uneq_smiles}

    return drug_name_2_pid, mult_uneq_smiles


def extract_pid_smiles(drug_name_to_pcomp_file, drug_smiles_file, force_run=True):
    if ((not os.path.exists(drug_smiles_file)) or force_run == True):
        with open(drug_name_to_pcomp_file, 'rb') as file:
            drug_name_to_pcomp = pickle.load(file)

        # drug_name_2_pid = {drug_name:drug_name_to_pcomp[drug_name].cid for drug_name in drug_name_to_pcomp }
        # drug_name_2_smiles = {drug_name:drug_name_to_pcomp[drug_name].canonical_smiles for drug_name in drug_name_to_pcomp}
        mult_uneq_smiles=find_drugs_with_ueq_smiles(drug_name_to_pcomp)
        #Now take the first pid and first smiles for each drug
        drug_name_2_pid = {drug_name: drug_name_to_pcomp[drug_name][0].cid for drug_name in
                           drug_name_to_pcomp if drug_name not in mult_uneq_smiles}
        drug_name_2_smiles = {drug_name:drug_name_to_pcomp[drug_name][0].canonical_smiles for drug_name
                              in drug_name_to_pcomp if drug_name not in mult_uneq_smiles}
        pid_df = pd.DataFrame({'drug_name': list(drug_name_2_pid.keys()), 'pid': list(drug_name_2_pid.values())}).set_index('drug_name')
        smiles_df = pd.DataFrame({'drug_name': list(drug_name_2_smiles.keys()), 'smiles': list(drug_name_2_smiles.values()) }).set_index('drug_name')
        drug_pid_smiles_df = pd.concat((pid_df, smiles_df), axis=1).reset_index()
        #save
        os.makedirs(os.path.dirname(drug_smiles_file), exist_ok=True)
        drug_pid_smiles_df.to_csv(drug_smiles_file, sep='\t', index=False)

    #TODO drop any duplicated pid-smiles row.
    drug_pid_smiles_df = pd.read_csv(drug_smiles_file, sep='\t', index_col=None)
    print('drugs (pids) with SMILES: ', len(drug_pid_smiles_df['pid'].unique()) )
    return drug_pid_smiles_df


def get_graph_from_smiles(drug_smiles_df, drug_graph_file, force_run=True):
    '''
    Given a dataframe with columns ['pid','smiles'], this function returns a dictionary with key=pid and
    value = (adjacency matrix, feature). Here, adjacency matrix and feature coming from a
    graph with atoms as nodes. We use rdkit and deepchem to extract graph and atom feature from
    SMILES respectively.
    '''
    if ((not os.path.exists(drug_graph_file)) or force_run == True):
        pid_to_smiles = zip(drug_smiles_df['pid'], drug_smiles_df['smiles'])
        pid_to_adjacency_mol_feat={}
        featurizer = dcm.feat.ConvMolFeaturizer()
        for pid, smiles in pid_to_smiles:
            mol = Chem.MolFromSmiles(smiles)
            mol_feat = featurizer.featurize(mol)
            #here, get_atom_features() returns a 2D numpy array of shape (#atoms_in_drug * #atom_features=75)
            #here get_adjacency_list() returns a list of lists with len=#atoms_in_drug
            '''from ConvMol, __init__(): atom_features: np.ndarray. Has shape (n_atoms, n_feat) adj_list:
             list List of length n_atoms, with neighbor indices of each atom.'''

            pid_to_adjacency_mol_feat[pid] = (mol_feat[0].get_atom_features(), mol_feat[0].get_adjacency_list())

        os.makedirs(os.path.dirname(drug_graph_file), exist_ok=True)
        with open(drug_graph_file, 'wb') as file:
            pickle.dump(pid_to_adjacency_mol_feat, file)

    with open(drug_graph_file, 'rb') as file:
        pid_to_adjacency_mol_feat = pickle.load(file)

    print('Get molecular graph for #pids: ', len(pid_to_adjacency_mol_feat.keys()))
    return pid_to_adjacency_mol_feat

def get_maccs(smiles):
    '''
    using RDKit
    :param smiles:
    :return: Return corresponding MACCS
    '''
    mol = Chem.MolFromSmiles(smiles)

    # Generate MACCS keys
    maccs_keys = MACCSkeys.GenMACCSKeys(mol)
    # print('MACCS', maccs_keys.ToBitString())
    return maccs_keys.ToBitString()

def get_morgan_fingerprint(smiles, dim):
    '''
    using RDKit
    :param smiles:
    :return: Return corresponding Morgan fingerprint
    '''
    mol = Chem.MolFromSmiles(smiles)

    # Compute the Morgan fingerprint with chirality information
    #According to PRODeepSyn by Wang et al.
    # (Link: https://academic.oup.com/bib/article/23/2/bbab587/6511206#338436757)
    # using radius=2 and nbits=256.
    mfp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=dim)
    mfp = list(np.array(mfp))
    # print('MFP: ', mfp)
    return mfp

def get_ecfp_4(smiles, rad, dim):
    '''
    using RDKit
    :param smiles:
    :return: Return corresponding ECFP_4
    '''
    mol = Chem.MolFromSmiles(smiles)
    # Compute the ECFP-6 fingerprint
    #According to A systematic evaluation of deep learning methods for the prediction of drug synergy in cancer, Baptista et al
    #using radius=2 and nbits=1024
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=rad, nBits=dim)
    # Convert the fingerprint to a numpy array
    ecfp6_vector = list(np.array(ecfp))
    # print('ECFP: ', ecfp6_vector)
    return ecfp6_vector

# def get_dmf(smiles):
#     '''
#     using ChemmineR
#     :param smiles:
#     :return: Return corresponding DMF
#     '''


def get_maccs_from_smiles(drug_smiles_df, maccs_file , force_run=True ):
    '''
    param drug_smiles_df: Given dataframe with columns ['pid','smiles']
    param drug_chemprop_file:   save extracted properties of drug as dataframes in the given directory.
    '''
    if (not os.path.exists(maccs_file)) or (force_run):
        pid_to_smiles = zip(drug_smiles_df['pid'], drug_smiles_df['smiles'])
        maccs = []

        pids = []
        for pid, smiles in pid_to_smiles:
            if 'MACCS' in properties:
                # extract MACCS using RDKit
                maccs.append(get_maccs(smiles))
            pids.append(pid)

        os.makedirs(os.path.dirname(maccs_file), exist_ok=True)

        #save MACCS
        if 'MACCS' in properties:
            maccs_df = pd.DataFrame({'pid': pids, 'MACCS':maccs }).set_index('pid')
            maccs_df = maccs_df['MACCS'].apply(list).apply(pd.Series)
            # Rename the columns to meaningful names (e.g., 'bit_0', 'bit_1', etc.)
            new_column_names = [f'MACCS_{i}' for i in range(len(maccs_df.columns))]
            maccs_df.columns = new_column_names
            maccs_df.reset_index().to_csv(maccs_file, sep='\t', index=False)

    if 'MACCS' in properties:
        maccs_df = pd.read_csv(maccs_file, sep='\t', index_col=None)
    print('Extracted MACCS from SMILES')
    return maccs_df


def get_mfp_from_smiles(drug_smiles_df,mfp_file,m_dim=256, force_run=True ):
    '''
    param drug_smiles_df: Given dataframe with columns ['pid','smiles']
    param drug_chemprop_file:   save extracted properties of drug as dataframes in the given directory.
    '''

    # mfp_file = drug_chemprop_dir + 'Morgan_fingerprint.tsv'
    # m_dim=256
    if not (os.path.exists(mfp_file)) or (force_run):
        pid_to_smiles = zip(drug_smiles_df['pid'], drug_smiles_df['smiles'])
        morgan_fp = []
        pids = []
        for pid, smiles in pid_to_smiles:
            if 'MFP' in properties:
                # extract Morgan fingerprint using RDKit
                morgan_fp.append(get_morgan_fingerprint(smiles, m_dim))
            pids.append(pid)

        os.makedirs(os.path.dirname(mfp_file), exist_ok=True)

        #save Morgan fingerprint
        if 'MFP' in properties:
            mfp_df = pd.DataFrame({'pid': pids,'Morgan_FP': morgan_fp}).set_index('pid')
            # Explode the 'Morgan_FP' column
            mfp_df = mfp_df['Morgan_FP'].apply(pd.Series)
            # Rename the new columns
            feature_columns = [f'Morgan_FP_{i}' for i in range(m_dim)]
            mfp_df.columns = feature_columns
            mfp_df.reset_index().to_csv(mfp_file, sep='\t', index=False)

    if 'MFP' in properties:
        mfp_df = pd.read_csv(mfp_file, sep='\t', index_col=None)

    print('Extracted MFP from SMILES')
    return mfp_df


# def get_ecfp_from_smiles(drug_smiles_df, ecfp_file,e_dim=1024, e_rad=2, force_run=True ):
#     '''
#     param drug_smiles_df: Given dataframe with columns ['pid','smiles']
#     param drug_chemprop_file:   save extracted properties of drug as dataframes in the given directory.
#     '''
#     # ecfp_file = drug_chemprop_dir + 'ECFP_4.tsv'
#     # e_dim=1024
#     # e_rad = 2
#     if ( not(os.path.exists(ecfp_file))) or (force_run == True):
#         pid_to_smiles = zip(drug_smiles_df['pid'], drug_smiles_df['smiles'])
#         ecfp_4 = []
#         pids = []
#         for pid, smiles in pid_to_smiles:
#             if 'ECFP_4' in properties:
#                 # extract ECFP_4 using RDKit
#                 ecfp_4.append(get_ecfp_4(smiles,e_rad, e_dim))
#             # extract drug molecular fingerprints (DMF) using ChemmineR
#             # dmf.append(get_dmf(smiles))
#             pids.append(pid)
#
#         os.makedirs(os.path.dirname(ecfp_file), exist_ok=True)
#
#         # save ECFP_4
#         if 'ECFP_4' in properties:
#             ecfp_df = pd.DataFrame({'pid': pids,'ECFP4': ecfp_4}).set_index('pid')
#             # Explode the 'Morgan_FP' column
#             ecfp_df = ecfp_df['ECFP4'].apply(pd.Series)
#             # Rename the new columns
#             feature_columns = [f'ECFP4_{i}' for i in range(e_dim)]
#             ecfp_df.columns = feature_columns
#             ecfp_df.reset_index().to_csv(ecfp_file, sep='\t', index=False)
#
#
#     if 'ECFP_4' in properties:
#         ecfp_4_df = pd.read_csv(ecfp_file, sep='\t', index_col=None)
#     print('Extracted ecfp_4_df from SMILES')
#     return ecfp_4_df


def get_chemprop_from_smiles(drug_smiles_df,properties, drug_chemprop_dir, force_run=True ):
    '''
    param drug_smiles_df: Given dataframe with columns ['pid','smiles']
    param drug_chemprop_file:   save extracted properties of drug as dataframes in the given directory.
    '''

    if 'MACCS' in properties:
        maccs_file = drug_chemprop_dir + 'MACCS.tsv'
    if 'MFP' in properties:
        mfp_file = drug_chemprop_dir + 'Morgan_fingerprint.tsv'
        m_dim=256
    if 'ECFP_4' in properties:
        ecfp_file = drug_chemprop_dir + 'ECFP_4.tsv'
        e_dim=1024
        e_rad = 2
    if ((not os.path.exists(maccs_file) or not (os.path.exists(mfp_file)) or
         not(os.path.exists(ecfp_file))) or force_run == True):
        pid_to_smiles = zip(drug_smiles_df['pid'], drug_smiles_df['smiles'])
        maccs = []
        morgan_fp = []
        ecfp_4 = []
        dmf= []
        pids = []
        for pid, smiles in pid_to_smiles:
            if 'MACCS' in properties:
                # extract MACCS using RDKit
                maccs.append(get_maccs(smiles))
            if 'MFP' in properties:
                # extract Morgan fingerprint using RDKit
                morgan_fp.append(get_morgan_fingerprint(smiles, m_dim))
            if 'ECFP_4' in properties:
                # extract ECFP_4 using RDKit
                ecfp_4.append(get_ecfp_4(smiles,e_rad, e_dim))
            # extract drug molecular fingerprints (DMF) using ChemmineR
            # dmf.append(get_dmf(smiles))
            pids.append(pid)

        os.makedirs(drug_chemprop_dir, exist_ok=True)

        #save MACCS
        if 'MACCS' in properties:
            maccs_df = pd.DataFrame({'pid': pids, 'MACCS':maccs }).set_index('pid')
            maccs_df = maccs_df['MACCS'].apply(list).apply(pd.Series)
            # Rename the columns to meaningful names (e.g., 'bit_0', 'bit_1', etc.)
            new_column_names = [f'MACCS_{i}' for i in range(len(maccs_df.columns))]
            maccs_df.columns = new_column_names

            maccs_df.reset_index().to_csv(maccs_file, sep='\t', index=False)

        #save Morgan fingerprint
        if 'MFP' in properties:
            mfp_df = pd.DataFrame({'pid': pids,'Morgan_FP': morgan_fp}).set_index('pid')
            # Explode the 'Morgan_FP' column
            mfp_df = mfp_df['Morgan_FP'].apply(pd.Series)
            # Rename the new columns
            feature_columns = [f'Morgan_FP_{i}' for i in range(m_dim)]
            mfp_df.columns = feature_columns
            mfp_df.reset_index().to_csv(mfp_file, sep='\t', index=False)

        # save ECFP_4
        if 'ECFP_4' in properties:
            ecfp_df = pd.DataFrame({'pid': pids,'ECFP4': ecfp_4}).set_index('pid')
            # Explode the 'Morgan_FP' column
            ecfp_df = ecfp_df['ECFP4'].apply(pd.Series)
            # Rename the new columns
            feature_columns = [f'ECFP4_{i}' for i in range(e_dim)]
            ecfp_df.columns = feature_columns
            ecfp_df.reset_index().to_csv(ecfp_file, sep='\t', index=False)


    drug_chemprop_dfs = {}
    if 'MACCS' in properties:
        maccs_df = pd.read_csv(maccs_file, sep='\t', index_col=None)
        drug_chemprop_dfs['MACCS'] = maccs_df
    if 'MFP' in properties:
        mfp_df = pd.read_csv(mfp_file, sep='\t', index_col=None)
        drug_chemprop_dfs['MFP'] = mfp_df
    if 'ECFP_4' in properties:
        ecfp_4_df = pd.read_csv(ecfp_file, sep='\t', index_col=None)
        drug_chemprop_dfs['ECFP_4'] = ecfp_4_df
    print('Extracted drug chemical properties from SMILES')
    return drug_chemprop_dfs


if __name__=='__main__':
    # if snakemake.params[0] == 'pid_map':
    #     mapped_syn_filename = snakemake.input[0]
    #     drug_name_to_pcomp_file = snakemake.output[0]
    #
    #     synergy_df = pd.read_csv(mapped_syn_filename, sep=',', low_memory=False,
    #                              dtype={'drug_row': str, 'drug_col': str, 'cell_line_name': str,
    #                                     'S_mean': float, 'synergy_zip': float})
    #     drug_names = list(set(synergy_df['drug_row']).union(set(synergy_df['drug_col'])))
    #     wrapper_download_pubchem_compound(drug_names, drug_name_to_pcomp_file)
    #
    # if snakemake.params[0]=='property':
    #     drug_name_to_pcomp_file = snakemake.input[0]
    #     drug_smiles_file = snakemake.output[0]
    #     drug_graph_file = snakemake.output[1]
    #     drug_chemprop_dir = snakemake.output[2]
    #     # extract and save drug_smiles
    #     drug_smiles_df = extract_pid_smiles(drug_name_to_pcomp_file, drug_smiles_file)
    #     # **************************************
    #
    #     # Extract graph from SMILES using deepchem
    #     pid_to_adjacency_mol_feat = get_graph_from_smiles(drug_smiles_df, drug_graph_file)
    #
    #     # Extract drug chemical properties using RDKit
    #     properties = ['MACCS', 'MFP', 'ECFP_4']
    #     pid_chemprop_dfs = get_chemprop_from_smiles(drug_smiles_df, properties, drug_chemprop_dir)
    drug_name_to_pcomp_file = '/home/grads/tasnina/Projects/SynVerse/inputs/drug/drug_name_to_pubchem_compound.pickle'
    drug_smiles_file = '/home/grads/tasnina/Projects/SynVerse/inputs/drug/smiles.tsv'
    drug_graph_file = '/home/grads/tasnina/Projects/SynVerse/inputs/drug/molecular_graph.pickle'
    drug_chemprop_dir = '/home/grads/tasnina/Projects/SynVerse/inputs/drug/chemprop/'
    # extract and save drug_smiles
    drug_smiles_df = extract_pid_smiles(drug_name_to_pcomp_file, drug_smiles_file)
    # **************************************

    # Extract graph from SMILES using deepchem
    pid_to_adjacency_mol_feat = get_graph_from_smiles(drug_smiles_df, drug_graph_file)

    # Extract drug chemical properties using RDKit
    properties = ['MACCS', 'MFP', 'ECFP_4']
    pid_chemprop_dfs = get_chemprop_from_smiles(drug_smiles_df, properties, drug_chemprop_dir)
