import numpy as np
import networkx as nx
import scipy.sparse as sp
import os
import pandas as pd

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def write_drug_drug_link_probability(pos_df, neg_df, run_, use_drug_feat_option, FLAGS, out_dir):
    lr = FLAGS.learning_rate
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    dr = FLAGS.dropout
    #inputs: df with link prediction probability after applying sigmoid on models predicted score for an edges(positive, negative)
    pos_out_file = out_dir + 'run_' + str(run_) + '/'+\
                    '/pos_val_scores'+'_drugfeat_'+str(use_drug_feat_option)+'_e_'+str(epochs) +'_lr_'+str(lr) +'_batch_'+ str(batch_size) +'_dr_'+\
                   str(dr)+'.tsv'
    neg_out_file = out_dir + 'run_' + str(run_) + '/'+\
                    '/neg_val_scores'+'_drugfeat_'+str(use_drug_feat_option)+'_e_'+str(epochs) +'_lr_'+str(lr) +'_batch_'+ str(batch_size) +'_dr_'+\
                   str(dr)+'.tsv'

    os.makedirs(os.path.dirname(pos_out_file), exist_ok=True)
    os.makedirs(os.path.dirname(neg_out_file), exist_ok=True)

    pos_df.to_csv(pos_out_file, sep='\t')
    neg_df.to_csv(neg_out_file, sep='\t')


def check_if_a_sparse_matrix_undirected(sp_matrix):

    #check if ppi matrix contains both (x,y) and (y,x) tuple
    sp_matrix_coo = sp_matrix.tocoo()
    row = sp_matrix_coo.row
    col = sp_matrix_coo.col
    x1 = set(tuple(zip(row,col)))
    x2 = set(tuple(zip(col,row)))
    if len(x1.union(x2)) == len(x1):
        # print('all (x,y)  and (y,x) tuple present')
        return 'undirected'
    else:
        # print('Not present')
        return 'directed'

def precision_at_k(y_true, y_score, k):
    df = pd.DataFrame({'true': y_true.tolist(), 'score': y_score.tolist()}).sort_values('score', ascending=False)
    df.reset_index(inplace=True)
    threshold = df.loc[k - 1]['score']
    df = df[df.score >= threshold]
    return df.true.sum() / df.shape[0]
