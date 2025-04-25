from rdkit import Chem

from preprocessing.rwr_runner import *
# from utils import *
from models.model_utils import *
from preprocessing.pretrain.embedding_generator import get_pretrained_embedding
import torch
from torch_geometric import data as DATA
from .preprocess_utils import adjacency_list_to_edges

def prepare_drug_features(drug_pids, params, inputs, device):
    dfeat_names = [f['name'] for f in params.drug_features]

    fields = ['norm','preprocess', 'encoder', 'value', 'compress', 'dim', 'use','file'] #for each feature we can have these fields.
    dfeat_dict = {field: {} for field in  fields}

    #parse norm, preprocessing and encoder for all features.
    dfeat_dict['preprocess'] = {f['name']: f.get('preprocess') for f in params.drug_features if f.get('preprocess') is not None}
    # dfeat_dict['filter'] = {f['name']: f.get('filter') for f in params.drug_features if f.get('filter') is not None}
    dfeat_dict['norm'] = {f['name']: f.get('norm') for f in params.drug_features if  f.get('norm') is not None}
    dfeat_dict['encoder'] = {f['name']: f.get('encoder') for f in params.drug_features if f.get('encoder') is not None}
    dfeat_dict['compress'] = {f['name']: f.get('compress', False) for f in params.drug_features}
    dfeat_dict['use'] = {f['name']: f.get('use') for f in params.drug_features}
    dfeat_dict['file'] = {f['name']: f.get('file') for f in params.drug_features if  f.get('file') is not None}



    if ('d1hot' in dfeat_names):
        one_hot_feat= pd.DataFrame(np.eye(len(drug_pids)))
        one_hot_feat['pid'] = drug_pids
        dfeat_dict['value']['d1hot'] = one_hot_feat
        dfeat_dict['dim']['d1hot'] = one_hot_feat.shape[1]-1

    if 'MACCS' in dfeat_names:
        maccs_file = inputs.maccs_file
        maccs_df = pd.read_csv(maccs_file,dtype={'pid':str}, sep='\t', index_col=None)

        dfeat_dict['value']['MACCS'] = maccs_df
        dfeat_dict['dim']['MACCS'] = maccs_df.shape[1]-1

    if 'MFP' in dfeat_names:
        mfp_file =  inputs.mfp_file
        mfp_df = pd.read_csv(mfp_file,dtype={'pid':str}, sep='\t', index_col=None)
        dfeat_dict['value']['MFP'] = mfp_df
        dfeat_dict['dim']['MFP'] = mfp_df.shape[1]-1

    if 'ECFP_4' in dfeat_names:
        ecfp_file = inputs.ecfp_file
        ecfp_df = pd.read_csv(ecfp_file,dtype={'pid':str}, sep='\t', index_col=None)
        dfeat_dict['value']['ECFP_4'] = ecfp_df
        dfeat_dict['dim']['ECFP_4'] = ecfp_df.shape[1]-1


    if 'mol_graph' in dfeat_names:
        mol_graph_file = inputs.mol_graph_file
        with open(mol_graph_file, 'rb') as file:
            pid_to_adjacency_mol_feat = pickle.load(file)
        #convert mol_graph to datatype compatible with GCN/GAT
        mol_pyg_dict, mol_feat_dim = mol_graph_to_GCN_data(pid_to_adjacency_mol_feat)
        dfeat_dict['value']['mol_graph'] = mol_pyg_dict
        dfeat_dict['dim']['mol_graph'] = mol_feat_dim

    if 'smiles' in dfeat_names:
        smiles_file = inputs.smiles_file
        smiles_df = pd.read_csv(smiles_file,dtype={'pid':str}, sep='\t', index_col=None)
        encoder_name = dfeat_dict['encoder'].get('smiles')
        # xx = smiles_df['smiles'].apply(lambda x: x[0:3])
        if encoder_name == 'Transformer':
            smiles_df, vocab_size = get_vocab_smiles(smiles_df)
            dfeat_dict['value']['smiles'] = smiles_df[['pid', 'tokenized']]
            dfeat_dict['dim']['smiles'] = vocab_size

        elif encoder_name == 'Transformer_Berttokenizer':
            dfeat_dict['value']['smiles'] = smiles_df[['pid', 'smiles']]
            dfeat_dict['file']['smiles'] = inputs.vocab_file

        elif encoder_name in ['SPMM', 'mole','kpgt']:
            embedding, embed_dim = get_pretrained_embedding(list(smiles_df['smiles']), params.input_dir,encoder_name, device)
            df = pd.DataFrame(embedding)
            df['pid'] = smiles_df['pid']
            dfeat_dict['value']['smiles'] = df
            dfeat_dict['dim']['smiles'] = embed_dim

        else:
            dfeat_dict['value']['smiles'] = smiles_df[['pid', 'smiles']]
            dfeat_dict['dim']['smiles'] = 0

    if 'target' in dfeat_names:
        target_file = inputs.target_file
        target_df = pd.read_csv(target_file,dtype={'pid':str, 'gene_name':str}, sep='\t', index_col=None)[['pid','gene_name']]
        target_df['pid'] = target_df['pid'].astype(str).apply(lambda x: x.replace('.0',''))

        #keep pids common between target_df and synergy_df
        target_df = target_df[target_df['pid'].isin(drug_pids)]
        target_feat_df = get_target_feat(target_df)

        if dfeat_dict['preprocess'].get('target')=='rwr':
            rwr_out_file = os.path.dirname(target_file) + '/rwr_target.tsv'
            rwr_target_feat_df = rwr_wrapper(target_feat_df, alpha=0.5, net_file=inputs.net_file, prot_info_file=inputs.prot_info_file, out_file=rwr_out_file, force_run=False)
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


def mol_graph_to_GCN_data(mol_graph_dict):
    '''convert atom features and adjacency list of each drug molecule into a data compatible with
    training pytorch geometric models'''
    mol_gcn_data_dict={}
    for pid in mol_graph_dict:
        mol_feat = mol_graph_dict[pid][0]
        mol_feat_dim = mol_feat.shape[1]
        c_size = mol_feat.shape[0]
        adj_list = mol_graph_dict[pid][1]

        edges = adjacency_list_to_edges(adj_list)
        GCNData = DATA.Data(x=torch.Tensor(mol_feat),
                    edge_index=torch.LongTensor(edges).transpose(1, 0)
                    if len(edges)>0 else torch.empty((2, 0), dtype=torch.long))
        GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
        mol_gcn_data_dict[str(pid)] = GCNData
    return mol_gcn_data_dict, mol_feat_dim
