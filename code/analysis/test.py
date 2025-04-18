import pandas as pd
import numpy as np
from utils import print_synergy_stat

score_names = ["S_mean_mean"]
synergy_file = f'/home/grads/tasnina/Projects/SynVerse/inputs/synergy/MARSY.csv'
synergy_df = pd.read_csv(synergy_file)


# synergy_file = f'/home/grads/tasnina/Projects/SynVerse/inputs/synergy/SynergyX.npy'
# data = np.load(synergy_file, allow_pickle=True)
# synergy_df = pd.DataFrame(data, columns=['drug_1_pid', 'drug_2_pid', 'cell_line_name', 'S_mean_mean']).\
#                     astype({'drug_1_pid':'string', 'drug_2_pid':'string', 'cell_line_name':'string', 'S_mean_mean':float})



#test if both (a,b) and (b,a) edges are present.
rev_pairs= set(zip(synergy_df['drug_1_pid'], synergy_df['drug_2_pid'])).intersection(set(zip(synergy_df['drug_2_pid'], synergy_df['drug_1_pid'])))

print(len(rev_pairs))