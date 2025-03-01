from network_algorithms.rwr import *
import pandas as pd
import os
import pickle
from scipy.sparse import csr_matrix

def prepare_network(net_file, prot_info_file, drug_target_df, confidence_threshold=900, force_run=False):

    W_out_file = f'{os.path.dirname(net_file)}/W_{confidence_threshold}.pickle'
    gene_idx_out_file = f'{os.path.dirname(net_file)}/gene_to_idx_{confidence_threshold}.pickle'

    if (os.path.exists(W_out_file) and os.path.exists(gene_idx_out_file) and (force_run==False)):
        with open(W_out_file, 'rb') as f:
            sparse_adj_matrix = pickle.load(f)
        with open(gene_idx_out_file, 'rb') as f:
            name_to_index = pickle.load(f)
        print('done loading network')

    else:
        #parse network
        net_df = pd.read_csv(net_file, sep=' ', compression='gzip')
        prot_info_df = pd.read_csv(prot_info_file, sep='\t', compression='gzip')

        merged_df = net_df.merge(prot_info_df[['#string_protein_id', 'preferred_name']],
                                 left_on='protein1',
                                 right_on='#string_protein_id',
                                 how='left').rename(columns={'preferred_name': 'source'})

        merged_df = merged_df.merge(prot_info_df[['#string_protein_id', 'preferred_name']],
                                    left_on='protein2',
                                    right_on='#string_protein_id',
                                    how='left').rename(columns={'preferred_name': 'target'})

        # Select only the required columns
        string_df = merged_df[['source', 'target', 'combined_score']]

        #filter out the low confidence edges
        string_df = string_df[string_df['combined_score']>confidence_threshold]
        genes = set(string_df['source']).union(string_df['target'])
        print(f'unique genes in STRING: {len(genes)}')
        print(f'edges in STRING: {len(string_df)}')
        # print(result_df.head(5))

        #convert pandas frame into numpy matrix
        string_genes = pd.unique(string_df[['source', 'target']].values.ravel('K'))

        #incporporate genes absent in STRING but present as drug target to the network
        target_genes = list(drug_target_df.columns)

        uncommon = set(target_genes).difference(set(string_genes))
        all_genes = list(set(target_genes).union(set(string_genes)))

        #create the adjacency matrix
        all_genes.sort()
        # Create a mapping from preferred names to indices
        name_to_index = {name: idx for idx, name in enumerate(all_genes)}

        # Initialize an empty adjacency matrix
        adj_matrix = np.zeros((len(all_genes), len(all_genes)), dtype=int)

        # Populate the adjacency matrix
        for _, row in string_df.iterrows():
            i = name_to_index[row['source']]
            j = name_to_index[row['target']]
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # Since the graph is undirected
        print('done preparing network')

        sparse_adj_matrix = csr_matrix(adj_matrix)

        with open(W_out_file, 'wb') as f:
            pickle.dump(sparse_adj_matrix, f)
        with open(gene_idx_out_file, 'wb') as f:
            pickle.dump(name_to_index, f)

    return sparse_adj_matrix, name_to_index

def rwr_wrapper(drug_target_df, net_file=None, prot_info_file=None, alpha=0.5, out_file=None, force_run=True):
    '''
    :param W: PPI network
    :param positives: the seed nodes
    :param alpha: restart parameter
    :return:
    '''

    variance_threshold = 0.0000001
    if (os.path.exists(out_file) and not force_run):
        rwr_score_df = pd.read_csv(out_file,dtype={'pid':str}, sep='\t', index_col=None)
    else:
        drug_target_df.set_index('pid', inplace=True)
        W, gene_to_idx = prepare_network(net_file, prot_info_file, drug_target_df,
                                         confidence_threshold=900, force_run=force_run)

        rwr_score={}
        for i, row in drug_target_df.iterrows():
            positive_weights = {gene_to_idx[col]:
                                row[col] for col in drug_target_df.columns if row[col] == 1}
            rwr_score[i] = rwr(W, weights=positive_weights,
                            alpha=alpha, eps=0.01, maxIters=500,
                            verbose=False)
        drug_target_df.reset_index(inplace=True)
        rwr_score_df = pd.DataFrame.from_dict(rwr_score, orient='index').rename_axis('pid').reset_index()
        rwr_score_df.to_csv(out_file, sep='\t', index=False)

    #filter out the genes with 0 variance.
    rwr_score_df.set_index('pid', inplace=True)
    rwr_score_df = rwr_score_df.loc[:, rwr_score_df.var()>variance_threshold]
    rwr_score_df.reset_index(inplace=True)

    return rwr_score_df
