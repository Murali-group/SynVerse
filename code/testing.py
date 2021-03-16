import sys
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import time
import random
# x = [[1,2],[3,4],[5,6]]
# y = [[3,4],[9,10],[11,12]]
#
# z = [[a,b] for a, b in x]
# w = [[b,a] for a, b in x]
# k= z + w
#
# x = [[1,2],[3,4],[5,6]]
# m = set(map(tuple, x))
# # n = [[a,b] for a, b in x]
#
# print(n)


#
# df = pd.DataFrame(np.array([[10, 20, 30], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
# print(df)
# print(df.index)
# print(list(df.index))


# row = np.array([0, 0, 1, 3, 3, 2])
# col = np.array([0, 2, 2, 0, 1, 2])
# col2 = np.array([0, 2, 5, 0, 1, 2])
# y = list(zip(row,col,col2))
#
# z = [(x1,x2) for x1, x2,x3 in y if x3==5]
# print(z)
# data = np.array([1, 2, 3, 4, 5, 6])
# csr_mat = csr_matrix((data, (row, col)), shape=(4, 4))
#
# mat = csr_mat.tocoo()
# row = mat.row
# col =  mat.col
#
# x = set(tuple(zip(row,col)))
# y = set(tuple(zip(col,row)))
# print(x)
# print(y)
# print(x.union(y))
#
# x = [[1,2],[5,4],[5,6]]
# # m = set(map(tuple, x))
# # edges_directed = [[idx_1, idx_2] if idx_1>idx_2 else None for idx_1, idx_2 in m]
# # edges_directed = list(filter(None, edges_directed))
# # print(edges_directed)
# y=list(np.array(x))
# z = random.sample(x, 2)
# print(z)

df = pd.DataFrame({'drug_1':[], 'drug_2':[]})
