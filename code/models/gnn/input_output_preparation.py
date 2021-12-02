import torch
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import os
import models.utils as model_utils
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


################################          PREPARE INPUT FOR TRAINING THE MODEL            #####################
def prepare_drug_feat(drug_maccs_keys_feature_df, drug_node_2_idx, n_drugs, use_drug_feat_option):
    if use_drug_feat_option:
        drug_maccs_keys_feature_df['drug_idx'] = drug_maccs_keys_feature_df['pubchem_cid']. \
            apply(lambda x: drug_node_2_idx[x])
        drug_maccs_keys_feature_df = drug_maccs_keys_feature_df.sort_values(by=['drug_idx'])
        assert len(drug_maccs_keys_feature_df) == n_drugs, 'problem in drug feat creation'
        drug_feat = torch.tensor(drug_maccs_keys_feature_df.drop(columns=['pubchem_cid']).set_index('drug_idx').to_numpy())
        # drug_num_feat = drug_feat.shape[1]
        # drug_nonzero_feat = np.count_nonzero(drug_feat)
        #
        # drug_feat = sp.csr_matrix(drug_feat)
        # drug_feat = utils.sparse_to_tuple(drug_feat.tocoo())

    else:
        # #one hot encoding for drug features
        # drug_maccs_keys_feature_df['drug_idx'] = drug_maccs_keys_feature_df['pubchem_cid']. \
        #     apply(lambda x: drug_node_2_idx[x])
        drug_feat = torch.tensor(np.identity(n_drugs))

    return drug_feat

def create_gene_gene_network(ppi,gene_node_2_idx=None,idx_2_cell_line=None, cell_line_2_tissue_dict=None, total_cell_lines=None):

    ''' convert ppi (if it is a sparse matrix) or each element of ppi (if it is dict of ppi_dfs containing protein-protein edges) into adjacency matrix'''
    gene_gene_adj_list = []
    if type(ppi) is dict:

        for tissue in ppi:
            ppi_df = ppi[tissue]
            ppi_df['p1'] = ppi_df['p1'].astype(str).apply(lambda x: gene_node_2_idx[x])
            ppi_df['p2'] = ppi_df['p2'].astype(str).apply(lambda x: gene_node_2_idx[x])

        for cell_line_idx in range(total_cell_lines):
            #     print('cell_line_idx',cell_line_idx)
            cell_line_name = idx_2_cell_line[cell_line_idx]
            tissue_name = cell_line_2_tissue_dict[cell_line_name]

            ppi_df = ppi[tissue_name]
            edges = list(zip(ppi_df['p1'], ppi_df['p2']))

            n_genes = len(gene_node_2_idx.keys())
            mat = np.zeros((n_genes, n_genes))
            for p1, p2 in edges:
                mat[p1, p2] = mat[p2, p1] = 1.

            gene_gene_adj_list.append(sp.csr_matrix(mat))
            print(cell_line_name, cell_line_idx)

    else:
        gene_gene_adj_list.append(nx.adjacency_matrix(nx.convert_matrix.\
                        from_scipy_sparse_matrix(ppi, create_using=nx.Graph(), edge_attribute=None)))

    return gene_gene_adj_list


def create_drug_target_network(drug_target_df,gene_node_2_idx,drug_node_2_idx):

    # now create drug_target adjacency matrix where gene nodes are in same order as they are in gene_net
    drug_target_adj_list=[]
    target_drug_adj_list = []

    row = list(drug_target_df['drug_idx'])
    col = list(drug_target_df['gene_idx'])
    data = np.ones(len(row))
    drug_target_adj = sp.csr_matrix((data, (row, col)), shape=(len(drug_node_2_idx.keys()), len(gene_node_2_idx.keys())))
    drug_target_adj_list.append(drug_target_adj)

    target_drug_adj_list.append(drug_target_adj.transpose(copy=True))

    return drug_target_adj_list, target_drug_adj_list

def create_drug_drug_network(synergy_df,n_drugs,total_cell_lines):
    drug_drug_adj_list = []
    for cell_line_idx in range(total_cell_lines):
        #     print('cell_line_idx',cell_line_idx)
        df = synergy_df[synergy_df['Cell_line_idx'] == cell_line_idx][['Drug1_idx', 'Drug2_idx',
                                                                       'Cell_line_idx', 'Loewe_label']]
        edges = list(zip(df['Drug1_idx'], df['Drug2_idx']))

        mat = np.zeros((n_drugs, n_drugs))
        for d1, d2 in edges:
            mat[d1, d2] = mat[d2, d1] = 1.
        #         print(d1,d2)
        drug_drug_adj_list.append(sp.csr_matrix(mat))

    return drug_drug_adj_list



def get_set_containing_all_folds(cross_validation_folds_pos_drug_drug_edges):
    all_folds_edges = set()
    for fold,edges in cross_validation_folds_pos_drug_drug_edges.items():
        all_folds_edges = all_folds_edges.union(set(edges))
    return all_folds_edges

def prepare_train_edges(cross_validation_folds_pos_drug_drug_edges, cross_validation_folds_neg_drug_drug_edges, \
                    cross_validation_folds_pos_non_drug_drug_edges, cross_validation_folds_neg_non_drug_drug_edges,
                    fold_no, total_cell_lines):
    #this returns four dictionaries. one dict for each of train_pos, train_neg, val_pos, val_neg.
    # in each dictionary: key = edge_type i.e. ( 'gene_gene', 'target_drug', 'drug_target', 'drug_drug')
    # each key maps to a list of torchtensor. e.g. drug_drug key maps to a list of torch tensor where each tensor is for a cell line


    edge_types = ['gene_gene', 'target_drug', 'drug_target', 'drug_drug']
    train_pos_edges_dict = {edge_type: [] for edge_type in edge_types}
    train_neg_edges_dict = {edge_type: [] for edge_type in edge_types}

    val_pos_edges_dict = {edge_type: [] for edge_type in edge_types}
    val_neg_edges_dict = {edge_type: [] for edge_type in edge_types}

    val_es_pos_edges_dict = {edge_type: [] for edge_type in edge_types}
    val_es_neg_edges_dict = {edge_type: [] for edge_type in edge_types}

    test_pos_edges_dict = {edge_type: [] for edge_type in edge_types}
    test_neg_edges_dict = {edge_type: [] for edge_type in edge_types}


    # set drug drug edges

    # all_folds_pos_edges = get_set_containing_all_folds(cross_validation_folds_pos_drug_drug_edges)
    train_pos_drug_drug_edges_set =  set(cross_validation_folds_pos_drug_drug_edges[fold_no]['train'])
    test_pos_drug_drug_edges_set =  set(cross_validation_folds_pos_drug_drug_edges[fold_no]['test'])
    val_pos_drug_drug_edges_set = set(cross_validation_folds_pos_drug_drug_edges[fold_no]['val'])
    val_es_pos_drug_drug_edges_set = set(cross_validation_folds_pos_drug_drug_edges[fold_no]['val_es'])

    train_neg_drug_drug_edges_set = set(cross_validation_folds_neg_drug_drug_edges[fold_no]['train'])
    test_neg_drug_drug_edges_set = set(cross_validation_folds_neg_drug_drug_edges[fold_no]['test'])
    val_neg_drug_drug_edges_set = set(cross_validation_folds_neg_drug_drug_edges[fold_no]['val'])
    val_es_neg_drug_drug_edges_set = set(cross_validation_folds_neg_drug_drug_edges[fold_no]['val_es'])

    for cell_line_idx in range(total_cell_lines):
        train_pos_source_nodes = [d1 for d1,d2,c in train_pos_drug_drug_edges_set if c == cell_line_idx]
        train_pos_target_nodes = [d2 for d1, d2, c in train_pos_drug_drug_edges_set if c == cell_line_idx]
        train_pos_edges = torch.stack([torch.LongTensor(train_pos_source_nodes),torch.LongTensor(train_pos_target_nodes)],dim=0).to(dev)

        train_neg_source_nodes = [d1 for d1, d2, c in train_neg_drug_drug_edges_set if c == cell_line_idx]
        train_neg_target_nodes = [d2 for d1, d2, c in train_neg_drug_drug_edges_set if c == cell_line_idx]
        train_neg_edges = torch.stack([torch.LongTensor(train_neg_source_nodes), torch.LongTensor(train_neg_target_nodes)], dim=0).to(dev)

        val_pos_source_nodes = [d1 for d1, d2, c in val_pos_drug_drug_edges_set if c == cell_line_idx]
        val_pos_target_nodes = [d2 for d1, d2, c in val_pos_drug_drug_edges_set if c == cell_line_idx]
        val_pos_edges = torch.stack(
            [torch.LongTensor(val_pos_source_nodes), torch.LongTensor(val_pos_target_nodes)], dim=0)

        val_neg_source_nodes = [d1 for d1, d2, c in val_neg_drug_drug_edges_set if c == cell_line_idx]
        val_neg_target_nodes = [d2 for d1, d2, c in val_neg_drug_drug_edges_set if c == cell_line_idx]
        val_neg_edges = torch.stack(
            [torch.LongTensor(val_neg_source_nodes), torch.LongTensor(val_neg_target_nodes)], dim=0)

        val_es_pos_source_nodes = [d1 for d1, d2, c in val_es_pos_drug_drug_edges_set if c == cell_line_idx]
        val_es_pos_target_nodes = [d2 for d1, d2, c in val_es_pos_drug_drug_edges_set if c == cell_line_idx]
        val_es_pos_edges = torch.stack(
            [torch.LongTensor(val_es_pos_source_nodes), torch.LongTensor(val_es_pos_target_nodes)], dim=0)

        val_es_neg_source_nodes = [d1 for d1, d2, c in val_es_neg_drug_drug_edges_set if c == cell_line_idx]
        val_es_neg_target_nodes = [d2 for d1, d2, c in val_es_neg_drug_drug_edges_set if c == cell_line_idx]
        val_es_neg_edges = torch.stack(
            [torch.LongTensor(val_es_neg_source_nodes), torch.LongTensor(val_es_neg_target_nodes)], dim=0)

        test_pos_source_nodes = [d1 for d1, d2, c in test_pos_drug_drug_edges_set if c == cell_line_idx]
        test_pos_target_nodes = [d2 for d1, d2, c in test_pos_drug_drug_edges_set if c == cell_line_idx]
        test_pos_edges = torch.stack(
            [torch.LongTensor(test_pos_source_nodes), torch.LongTensor(test_pos_target_nodes)], dim=0)

        test_neg_source_nodes = [d1 for d1, d2, c in test_neg_drug_drug_edges_set if c == cell_line_idx]
        test_neg_target_nodes = [d2 for d1, d2, c in test_neg_drug_drug_edges_set if c == cell_line_idx]
        test_neg_edges = torch.stack(
            [torch.LongTensor(test_neg_source_nodes), torch.LongTensor(test_neg_target_nodes)], dim=0)

        train_pos_edges_dict['drug_drug'].append(train_pos_edges)
        train_neg_edges_dict['drug_drug'].append(train_neg_edges)
        val_pos_edges_dict['drug_drug'].append(val_pos_edges)
        val_neg_edges_dict['drug_drug'].append(val_neg_edges)
        val_es_pos_edges_dict['drug_drug'].append(val_es_pos_edges)
        val_es_neg_edges_dict['drug_drug'].append(val_es_neg_edges)
        test_pos_edges_dict['drug_drug'].append(test_pos_edges)
        test_neg_edges_dict['drug_drug'].append(test_neg_edges)

    #NO TEST DATASET for non drug-drug edges yet
    #gene_gene edges, target_drug_edges, drug_target_edges

    non_drug_drug_edge_types = ['gene_gene', 'target_drug', 'drug_target']

    for edge_type in non_drug_drug_edge_types:

        val_pos_non_drug_drug_edges_set = set(cross_validation_folds_pos_non_drug_drug_edges[edge_type][fold_no])
        all_folds_pos_edges = get_set_containing_all_folds(cross_validation_folds_pos_non_drug_drug_edges[edge_type])
        train_pos_non_drug_drug_edges_set = all_folds_pos_edges.difference(val_pos_non_drug_drug_edges_set)

        val_neg_non_drug_drug_edges_set = set(cross_validation_folds_neg_non_drug_drug_edges[edge_type][fold_no])
        all_folds_neg_edges = get_set_containing_all_folds(cross_validation_folds_neg_non_drug_drug_edges[edge_type])
        train_neg_non_drug_drug_edges_set = all_folds_neg_edges.difference(val_neg_non_drug_drug_edges_set)


        train_pos_source_nodes = [d1 for d1, d2 in train_pos_non_drug_drug_edges_set]
        train_pos_target_nodes = [d2 for d1, d2 in train_pos_non_drug_drug_edges_set]
        train_pos_edges = torch.stack(
            [torch.LongTensor(train_pos_source_nodes), torch.LongTensor(train_pos_target_nodes)], dim=0).to(dev)

        train_neg_source_nodes = [d1 for d1, d2 in train_neg_non_drug_drug_edges_set]
        train_neg_target_nodes = [d2 for d1, d2 in train_neg_non_drug_drug_edges_set]
        train_neg_edges = torch.stack(
            [torch.LongTensor(train_neg_source_nodes), torch.LongTensor(train_neg_target_nodes)], dim=0).to(dev)

        val_pos_source_nodes = [d1 for d1, d2 in val_pos_non_drug_drug_edges_set]
        val_pos_target_nodes = [d2 for d1, d2 in val_pos_non_drug_drug_edges_set]
        val_pos_edges = torch.stack(
            [torch.LongTensor(val_pos_source_nodes), torch.LongTensor(val_pos_target_nodes)], dim=0)

        val_neg_source_nodes = [d1 for d1, d2 in val_neg_non_drug_drug_edges_set]
        val_neg_target_nodes = [d2 for d1, d2  in val_neg_non_drug_drug_edges_set]
        val_neg_edges = torch.stack(
            [torch.LongTensor(val_neg_source_nodes), torch.LongTensor(val_neg_target_nodes)], dim=0)

        train_pos_edges_dict[edge_type].append(train_pos_edges)
        train_neg_edges_dict[edge_type].append(train_neg_edges)
        val_pos_edges_dict[edge_type].append(val_pos_edges)
        val_neg_edges_dict[edge_type].append(val_neg_edges)

    return train_pos_edges_dict, train_neg_edges_dict, val_pos_edges_dict, val_neg_edges_dict,val_es_pos_edges_dict, val_es_neg_edges_dict, test_pos_edges_dict, test_neg_edges_dict,






################################          PREPARE OUTPUT FOR SAVING        #####################

def prepare_pred_score_for_saving(pos_pred, neg_pred, cell_line, idx_2_drug_node):
    pos_edge_dict = {'drug_1_idx': [], 'drug_2_idx': [], 'cell_line': [], 'predicted': [], 'true': []}
    neg_edge_dict = {'drug_1_idx': [], 'drug_2_idx': [], 'cell_line': [], 'predicted': [], 'true': []}


    pos_edge_dict['drug_1_idx'] = pos_pred[0].detach().cpu().numpy()
    pos_edge_dict['drug_2_idx'] = pos_pred[1].detach().cpu().numpy()
    pos_edge_dict['cell_line'] = np.array([cell_line]*pos_pred.size()[1])
    pos_edge_dict['predicted'] =  pos_pred[2].detach().cpu().numpy()
    pos_edge_dict['true'] =  pos_pred[3].detach().cpu().numpy()

    neg_edge_dict['drug_1_idx'] =  neg_pred[0].detach().cpu().numpy()
    neg_edge_dict['drug_2_idx'] = neg_pred[1].detach().cpu().numpy()
    neg_edge_dict['cell_line'] = np.array([cell_line]*neg_pred.size()[1])
    neg_edge_dict['predicted'] =  neg_pred[2].detach().cpu().numpy()
    neg_edge_dict['true'] = neg_pred[3].detach().cpu().numpy()

    pos_df = pd.DataFrame.from_dict(pos_edge_dict)
    neg_df = pd.DataFrame.from_dict(neg_edge_dict)

    pos_df['drug_1'] = pos_df['drug_1_idx'].apply(lambda x: idx_2_drug_node[x])
    pos_df['drug_2'] = pos_df['drug_2_idx'].apply(lambda x: idx_2_drug_node[x])

    neg_df['drug_1'] = neg_df['drug_1_idx'].apply(lambda x: idx_2_drug_node[x])
    neg_df['drug_2'] = neg_df['drug_2_idx'].apply(lambda x: idx_2_drug_node[x])

    pos_df = pos_df[['drug_1', 'drug_2', 'cell_line', 'predicted', 'true']].sort_values(by=['predicted'], \
                                                                                        ascending=False)
    neg_df = neg_df[['drug_1', 'drug_2', 'cell_line', 'predicted', 'true']].sort_values(by=['predicted'], \
                                                                                        ascending=False)
    return pos_df, neg_df


def save_best_model(best_model,param_dict, out_dir):
    #inputs: df with link prediction probability after applying sigmoid on models predicted score for an edges(positive, negative)

    param_str = model_utils.dict_to_str(param_dict)
    model_file = out_dir + 'model_file'+ param_str + '.p'

    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    torch.save(best_model.state_dict(), model_file)

    return model_file
