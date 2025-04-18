import pandas as pd
from utils import print_synergy_stat

feature_sets = ['D_ECFP_4_MACCS_MFP_d1hot_mol_graph_smiles_C_c1hot', 'D_d1hot_target_C_c1hot','D_d1hot_C_c1hot_genex_genex_lincs_1000']
# score_names = ["S_mean_mean"]
score_names = ["synergy_loewe_mean"]

for feature_set in feature_sets:
    for score_name in score_names:
        print(f'\n\n\n{feature_set}')
        synergy_file = f'/home/grads/tasnina/Projects/SynVerse/inputs/splits/{feature_set}/k_0.05_{score_name}/leave_comb_0.2_0.25/run_0_0/all.tsv'
        df = pd.read_csv(synergy_file, sep='\t')
        print_synergy_stat(df)
        # Count the number of rows per edge_type
        edge_counts = df['cell_line_name'].value_counts()

        # Edge type with the highest number of rows and its fraction
        highest_edge_type = edge_counts.idxmax()  # edge type with the highest count
        highest_count = edge_counts.max()  # highest count value
        fraction_highest = highest_count / len(df)  # fraction of triplets in the highest edge type

        print(f"highest: {highest_edge_type}  {highest_count}  {fraction_highest:.2f}")

        # Edge type with the lowest number of rows and its fraction
        lowest_edge_type = edge_counts.idxmin()  # edge type with the lowest count
        lowest_count = edge_counts.min()  # lowest count value
        fraction_lowest = lowest_count / len(df)  # fraction of triplets in the lowest edge type

        print(f"loewest count: {lowest_edge_type}  {lowest_count}  {fraction_lowest:.2f}")


        # Mean and standard deviation of 'score'
        mean_score = df[score_name].mean()
        std_score = df[score_name].std()

        print(f"Mean of score: {mean_score:.2f}")
        print(f"Standard deviation of score: {std_score:.2f}")