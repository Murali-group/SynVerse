import pandas as pd
import numpy as np
import os
import pickle
from scipy.sparse import csr_matrix

net_file = '/home/grads/tasnina/Projects/SynVerse/datasets/network/STRING/9606.protein.links.v12.0.txt.gz'
confidence_threshold = 900
W_out_file = f'{os.path.dirname(net_file)}/W_{confidence_threshold}.pickle'
with open(W_out_file, 'rb') as f:
    sparse_adj_matrix = pickle.load(f)


print(f'STRING with threshold {confidence_threshold}: #nodes {sparse_adj_matrix.shape[0]}, '
          f'#edges: {sparse_adj_matrix.nnz}')
print('done loading network')