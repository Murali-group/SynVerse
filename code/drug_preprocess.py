from network_algorithms.rwr_runner import *
from utils import *
from models.model_utils import *
from models.pretrained.embedding_generator import get_SPMM_embedding, get_mole_embedding

def prepare_drug_features(drug_features, drug_pids, params, inputs, device):
    dfeat_names = [f['name'] for f in drug_features]

    fields = ['norm','preprocess','filter', 'encoder', 'value', 'compress', 'dim', 'use'] #for each feature we can have these fields.
    dfeat_dict = {field: {} for field in  fields}

    #parse norm, preprocessing and encoder for all features.
    dfeat_dict['preprocess'] = {f['name']: f.get('preprocess') for f in drug_features if f.get('preprocess') is not None}
    dfeat_dict['filter'] = {f['name']: f.get('filter') for f in drug_features if f.get('filter') is not None}
    dfeat_dict['norm'] = {f['name']: f.get('norm') for f in drug_features if  f.get('norm') is not None}
    dfeat_dict['encoder'] = {f['name']: f.get('encoder') for f in drug_features if f.get('encoder') is not None}
    dfeat_dict['compress'] = {f['name']: f.get('compress', False) for f in drug_features}
    dfeat_dict['use'] = {f['name']: f.get('use') for f in drug_features}


    if ('d1hot' in dfeat_names):
        one_hot_feat= pd.DataFrame(np.eye(len(drug_pids)))
        one_hot_feat['pid'] = drug_pids
        dfeat_dict['value']['d1hot'] = one_hot_feat
        dfeat_dict['dim']['d1hot'] = one_hot_feat.shape[1]-1

    if 'MACCS' in dfeat_names:
        chem_prop_dir = params.drug_chemprop_dir
        maccs_file = chem_prop_dir + 'MACCS.tsv'
        maccs_df = pd.read_csv(maccs_file,dtype={'pid':str}, sep='\t', index_col=None)

        #TODO: if any preprocessing step is mentioned for 'MACCS' feature, do that here.
        dfeat_dict['value']['MACCS'] = maccs_df
        dfeat_dict['dim']['MACCS'] = maccs_df.shape[1]-1

    if 'MFP' in dfeat_names:
        chem_prop_dir = params.drug_chemprop_dir
        mfp_file = chem_prop_dir + 'Morgan_fingerprint.tsv'
        mfp_df = pd.read_csv(mfp_file,dtype={'pid':str}, sep='\t', index_col=None)
        #TODO: if any preprocessing step is mentioned for 'MACCS' feature, do that here.
        dfeat_dict['value']['MFP'] = mfp_df
        dfeat_dict['dim']['MFP'] = mfp_df.shape[1]-1

    if 'ECFP_4' in dfeat_names:
        chem_prop_dir = params.drug_chemprop_dir
        ecfp_file = chem_prop_dir + 'ECFP_4.tsv'
        ecfp_df = pd.read_csv(ecfp_file,dtype={'pid':str}, sep='\t', index_col=None)
        #TODO: if any preprocessing step is mentioned for 'MACCS' feature, do that here.
        dfeat_dict['value']['ECFP_4'] = ecfp_df
        dfeat_dict['dim']['ECFP_4'] = ecfp_df.shape[1]-1


    if 'mol_graph' in dfeat_names:
        mol_graph_file = inputs.drug_graph_file
        with open(mol_graph_file, 'rb') as file:
            pid_to_adjacency_mol_feat = pickle.load(file)
        #convert mol_graph to datatype compatible with GCN/GAT
        mol_pyg_dict, mol_feat_dim = mol_graph_to_GCN_data(pid_to_adjacency_mol_feat)
        dfeat_dict['value']['mol_graph'] = mol_pyg_dict
        dfeat_dict['dim']['mol_graph'] = mol_feat_dim

    if 'smiles' in dfeat_names:
        smiles_file = inputs.drug_smiles_file
        smiles_df = pd.read_csv(smiles_file,dtype={'pid':str}, sep='\t', index_col=None)

        if dfeat_dict['encoder'].get('smiles') == 'Transformer':
            smiles_df, vocab_size = get_vocab_smiles(smiles_df)
            dfeat_dict['value']['smiles'] = smiles_df[['pid', 'tokenized']]
            dfeat_dict['dim']['smiles'] = vocab_size

        elif dfeat_dict['encoder'].get('smiles') == 'SPMM':
            embedding, embed_dim = get_SPMM_embedding(list(smiles_df['smiles']), params.input_dir, device)
            spmm_df = pd.DataFrame(embedding)
            spmm_df['pid'] = smiles_df['pid']
            dfeat_dict['value']['smiles'] = spmm_df
            dfeat_dict['dim']['smiles'] = embed_dim

        elif dfeat_dict['encoder'].get('smiles') == 'mole':
            embedding, embed_dim = get_mole_embedding(list(smiles_df['smiles']), params.input_dir)
            mole_df = pd.DataFrame(embedding)
            mole_df['pid'] = smiles_df['pid']
            dfeat_dict['value']['smiles'] = mole_df
            dfeat_dict['dim']['smiles'] = embed_dim


    if 'target' in dfeat_names:
        target_file = inputs.drug_target_file
        target_df = pd.read_csv(target_file,dtype={'pid':str, 'gene_name':str}, sep='\t', index_col=None)[['pid','gene_name']]
        target_df['pid'] = target_df['pid'].astype(str).apply(lambda x: x.replace('.0',''))

        #keep pids common between target_df and synergy_df
        target_df = target_df[target_df['pid'].isin(drug_pids)]
        target_feat_df = get_target_feat(target_df)

        if dfeat_dict['preprocess'].get('target')=='rwr':
            rwr_out_file = os.path.dirname(target_file) + '/rwr_target.tsv'
            rwr_target_feat_df = rwr_wrapper(target_feat_df, alpha=0.5, out_file=rwr_out_file, force_run=False)
            dfeat_dict['value']['target']= rwr_target_feat_df
            dfeat_dict['dim']['target']= rwr_target_feat_df.shape[1] - 1

        else:
            dfeat_dict['value']['target'] = target_feat_df
            dfeat_dict['dim']['target'] = target_feat_df.shape[1] - 1

    return dfeat_dict, dfeat_names


def get_target_feat(target_df):
    '''
    This code will create a dataframe where each row represents a pid and the columns are
    the unique uniprot_id values. The cells contain 1 if the uniprot_id is associated with the pid, and 0 otherwise.
    '''
    # Create the pivot table
    # TODO: uncomment the following once target data parsing is done.
    target_feat_df = target_df.pivot_table(index='pid', columns='gene_name', aggfunc='size', fill_value=0)
    # target_feat_df = target_df.pivot_table(index='pid', columns='uniprot_id', aggfunc='size', fill_value=0)

    # Ensure the values are only 1 or 0
    target_feat_df = target_feat_df.applymap(lambda x: 1 if x > 0 else 0)

    # Reset the index to make 'pid' a column again (optional)
    target_feat_df.reset_index(inplace=True)
    return target_feat_df

