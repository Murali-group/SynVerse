from graph_utils import rewire_signed
import pandas as pd

df = pd.DataFrame({'source': ['g1','g2', 'pink', 'yellow', 'g1', 'g1' ],
    'target': ['g2', 'pink', 'yellow', 'purple', 'purple', 'blue'],
     'edge_type': [1, 1, 1, 1,1,1],
     'score': [1, 1, 10, 5, 10, 1]})

rewired_df = rewire_signed(df, 'score', method='SA')
print(df[['source', 'target', 'score']])
print(rewired_df[['source', 'target', 'score']])