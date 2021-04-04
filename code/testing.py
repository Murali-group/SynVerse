import sys
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import time
import random
import scipy.sparse as sp
import matplotlib.pyplot as plt
import re
import statistics

# x=[0, 0, 1, 3, 3,5,6,7,8, 20]
# y=[1,2]
# x+=y
# print(type(x))
# print(x)

# x = random.choices([10,20,30,40,50], k=10)
# y = random.choices([50,60,70,80,90], k=10)
# z = list(zip(x,y))
# print(z)

# z = list(z)[0:8]
# z = np.array(z)
# df = pd.DataFrame({'drug_1': z[:,0], 'drug_2': z[:, 1]})
#
# # print(z)
# print(list(df.index))
#
# # m = 'hello'
# # r = [m]*10
# # print(r)

#
# df = pd.DataFrame([[1, 2, 3], [4, 5, 5], [4, 6, 1]], columns = ["a", "b", "c"])
# df['abmax'] =df[['a','b']].max(axis=1)
# print(df)
# df1 = pd.DataFrame(np.array([[1, 2, 0], [2, 5, 6], [0, 8, 9]]), columns=['a', 'b', 'c'])
# print(list(df.index))
# df = pd.concat([df, df1], axis=0)
# df.reset_index(drop=True, inplace=True)
# print(list(df.index))
# print(df)

# print(df)
# print(df1)
#
# df_val = df.values
# df1_val = df1.values
# print(type(df_val))
# x = np.concatenate((df_val, df1_val), axis=1)
# y = np.concatenate((df1_val, df_val), axis=1)
# z = np.concatenate((x,y), axis=0)
#
# print(x)
# print(y)
# print(z)
# df = df.reindex(df1['d']).reset_index()


# df = df.sort_values(["b", "c"], ascending = (False, False))
# print(df)


# x=[0, 0, 1, 3, 3,5,6,7,8, 20]
# print(statistics.median(x))
# y=[3,5,6,7, 2]
# data = [x,y]
# print(type(x))
# plt.boxplot(data,labels=['hello1','hello2'])
# plt.xticks(rotation='vertical', fontsize=10)
# plt.margins(0.2)
# plt.subplots_adjust(bottom=0.15)
# plt.show()
# plt.clf()
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

# df = pd.DataFrame({'drug_1':[], 'drug_2':[]})
# drug_feat = sp.identity(5)
# drug_nonzero_feat, drug_num_feat = drug_feat.shape
# print(drug_nonzero_feat, drug_num_feat)

# x = [[1, 2], [3, 4], [5, 6]]
# y = set([(x1,x2) for x1,x2 in x])
# z = random.sample(y,2)
# print(y)
# print(z,type(z))
#
#
# df = pd.DataFrame([[1, 2, 3], [4, 5, 5], [4, 6, 1]], columns = ["a", "b", "c"])
# df = df.reindex([0,1,2,2,0,1])
# print(df)
#
# a = np.eye(3, dtype=int)
# df = pd.DataFrame(a)
# print(df)

# x=[1,2,3]
# y=[6,7,8]
# z=10
# x.append(z)
# print(x)

# x = {'a':[1,2], 'b':[4,5]}
# z = np.array(list(x.values()))
# print(z)
# print(z[:,0])
# print(len(z))
# print(np.arange(5))

# df1 = pd.DataFrame([[1, 2, 3], [1, 2, 5], [3, 60, 100]], columns = ["idx", "b", "c"])
# # df1.set_index('idx', inplace=True)
# # print(len(df1.columns))
# # df2 = pd.DataFrame([[2, 2, 3], [2, 2, 5], [3, 2, 100],[1,1,10],[1,2,40]], columns = ["idx_1", "idx_2", "c"])
# #
# # df3 = df1.reindex(df2['idx_1']).reset_index()
# # df4= df1.reindex(df2['idx_2']).reset_index()
# #
# #
# # print(df1)
# # print(df2)
# # print(df3)
# # print(df4)
# #
# # df = pd.concat([df3,df4], axis=1)
# # print(df)
# #
# # df.drop(['idx_1','idx_2'], axis=1, inplace=True)
# # print(df)
# df1.set_index(['idx','b'], inplace=True)
# print(df1)
# # x = list(map(tuple,df1.index))
# x= [(x1,x2) for x1, x2 in df1.index]
# print(x)

#

# train_edges = np.array([[1],[3]])
# train_edges = list(train_edges.squeeze(axis=1))
#
# print(train_edges)
# # print(train_edges[:, 0], train_edges[:, 1])


# x = {'a':1, 'kb':2, 'c': 20}
# plt.scatter(x.keys(), x.values())
# plt.show()
# df = pd.DataFrame([[1, 2, 3], [1, 2, 5], [3, 60, 100]], columns = ["idx", "b", "c"])
# x = list(zip(df['c'],df['b']))
# x += [(x2,x1) for x1, x2 in x]
# print(x)

# data_dict = {'a': [1,2,3,4], 'b': [1,2,3], 'c': [2,45,67,93,82,92]}
# x = pd.DataFrame.from_dict(data_dict,orient='index').T
# print(x)

# posE = np.zeros((20,2))
# print(posE)
#
# from sklearn.model_selection import KFold
# cv = KFold(n_splits=5, random_state=24, shuffle=True)
# iCnt = 0
#
# for train_index, test_index in cv.split(posE):
#     print(train_index,test_index)
#     # if iCnt != 100:
#     #     train_posIdx = train_index
#     #     test_posIdx = test_index
#     #     break
#     iCnt +=1

# print(train_posIdx)
# print(test_posIdx)

x = [1,2,3,4]
y=[1,2,3,4, 7,8,9,10]
# print(random.choices(x, weights=[1,5,1,1],k=10))
z = list(zip(x,y))
print(z)