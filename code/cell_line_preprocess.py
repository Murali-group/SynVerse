#***************** CELL LINE Preprocess ***************************
import pandas as pd
import numpy as np
from utils import *

def prepare_cell_line_features(cell_line_features, cell_line_names,params, inputs):
    # cfeat_dict = {}
    # cfeat_dim_dict = {}
    # cfeat_names = [f['name'] for f in cell_line_features]
    # cfeat_preprocess = {f['name']: f['preprocess'] for f in cell_line_features}
    # cfeat_encoders = {f['name']: f.get('encoder') for f in cell_line_features}
    # cfeat_norm = {f['name']: f['norm'] for f in cell_line_features}

    cfeat_names = [f['name'] for f in cell_line_features]

    fields = ['norm', 'preprocess', 'encoder', 'mtx', 'dim','use']  # for each feature we can have these fields.
    cfeat_dict = {field: {} for field in fields}

    # parse norm, preprocessing and encoder for all features.
    cfeat_dict['preprocess'] = {f['name']: f.get('preprocess') for f in cell_line_features if f.get('preprocess') is not None}
    cfeat_dict['norm'] = {f['name']: f.get('norm') for f in cell_line_features if f.get('norm') is not None}
    cfeat_dict['encoder'] = {f['name']: f.get('encoder') for f in cell_line_features if f.get('encoder') is not None}
    cfeat_dict['use'] = {f['name']: f.get('use') for f in cell_line_features}


    if ('c1hot' in cfeat_names):
        one_hot_feat = pd.DataFrame(np.eye(len(cell_line_names)))
        one_hot_feat['cell_line_name'] = cell_line_names
        cfeat_dict['mtx']['c1hot'] = one_hot_feat
        cfeat_dict['dim']['c1hot'] = one_hot_feat.shape[1] - 1

    if 'genex' in cfeat_names:
        # feat_name = 'genex'
        ccle_file = inputs.cell_line_file
        ccle_df = pd.read_csv(ccle_file, sep='\t')

        # do any preprocessing
        if cfeat_dict['preprocess'].get('genex') == 'lincs_1000':
            # feat_name = feat_name + '_lincs_1000'
            ccle_df = landmark_gene_filter(ccle_df, inputs.lincs)
        # do any normalization
        if cfeat_dict['norm'].get('genex') is not None:
            # feat_name = feat_name + '_' + cfeat_norm['genex']
            ccle_df.set_index('cell_line_name', inplace=True)
            ccle_df, means, std = normalize(ccle_df, norm_type=cfeat_dict['norm'].get('genex'))
            ccle_df.reset_index(names='cell_line_name', inplace=True)

        cfeat_dict['mtx']['genex'] = ccle_df
        cfeat_dict['dim']['genex'] = ccle_df.shape[1] - 1

    return cfeat_dict, cfeat_names
def landmark_gene_filter(genex_df, lincs_file):
    landmark_genes_df = pd.read_csv(lincs_file,sep='\t', skiprows=lambda x: x == 1)[['L1000 Probe ID','Gene Entrez ID', 'Gene Symbol']]
    #remove the control/invariant genes/probes
    landmark_genes_df = landmark_genes_df[~landmark_genes_df['L1000 Probe ID'].str.contains('INV', case=True)]

    landmark_genes = list(set(landmark_genes_df['Gene Symbol']))

    filtered_genex_df = pd.concat([genex_df['cell_line_name'],genex_df[[col for col in genex_df.columns if col in landmark_genes]]], axis=1)
    print(len(filtered_genex_df.columns))
    print(len(genex_df.columns))
    return filtered_genex_df
