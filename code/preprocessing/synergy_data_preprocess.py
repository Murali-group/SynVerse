import pandas as pd
from utils import print_synergy_stat
def feature_based_filtering(synergy_df, dfeat_dict, cfeat_dict):
    '''
    If none of the features are optional, then we need to filter out the triplets such that only drugs and cell lines
    with all feature information available are in the final synergy triplets.
    '''
    drug_pids = set(synergy_df['drug_1_pid']).union(set(synergy_df['drug_2_pid']))
    cell_line_names = set(synergy_df['cell_line_name'])
    print('Before feature based fitering: ')
    print_synergy_stat(synergy_df)

    # find drugs with all features available
    for feat_name in dfeat_dict:
        if isinstance(dfeat_dict[feat_name],pd.DataFrame):
            drugs = set(dfeat_dict[feat_name]['pid'])
        elif isinstance(dfeat_dict[feat_name],dict):
            drugs = set(dfeat_dict[feat_name].keys())
        drug_pids = drug_pids.intersection(drugs)
        print(f'filtering for {feat_name}')


    # find cell lines with all features available
    for feat_name in cfeat_dict:
        cells = set(cfeat_dict[feat_name]['cell_line_name'])
        cell_line_names = cell_line_names.intersection(cells)

        print(f'filtering for {feat_name}')

    #filter synergy triplets
    synergy_df = synergy_df[(synergy_df['drug_1_pid'].isin(drug_pids)) & (synergy_df['drug_2_pid'].isin(drug_pids))
                            & (synergy_df['cell_line_name'].isin(cell_line_names)) ]

    # n_after_feat_filt =len(synergy_df)
    # if k>0: #keep only top k cell lines having the most synergy triplets.
    #     top_cell_lines = synergy_df['cell_line_name'].value_counts().nlargest(k).index
    #     print('top cell lines:' , top_cell_lines)
    #     synergy_df = synergy_df[synergy_df['cell_line_name'].isin(top_cell_lines)]
    #
    #     print(f'keeping top {k} cell lines, retrieved frac:{len(synergy_df)/n_after_feat_filt}')
    #assert that there is no duplicate triplets in synergy_df
    triplets = list(zip(synergy_df['drug_1_pid'],synergy_df['drug_2_pid'],synergy_df['cell_line_name']))
    assert len(set(triplets))==len(triplets), print('still some duplicates remaining')

    print('After feature based filtering: ')
    print_synergy_stat(synergy_df)

    return synergy_df


def abundance_based_filtering(synergy_df, min_frac=0):
    # Initialize the filtered_df as empty
    filtered_df = pd.DataFrame()

    # Sort cell lines by their count in descending order
    cell_line_counts = synergy_df['cell_line_name'].value_counts()

    #option 1: min frac samples present from each cell line in the final dataset. More balanced data.
    for cell_line, count in cell_line_counts.items():
        # Calculate the potential new total rows if this cell line is added
        new_total_rows = len(filtered_df) + count
        new_fraction = count / new_total_rows

        # If adding this cell line meets the threshold, add it to filtered_df
        if new_fraction >= min_frac:
            filtered_df = pd.concat([filtered_df, synergy_df[synergy_df['cell_line_name'] == cell_line]])


    print('After abundance based filtering: ')
    print('frac triplets retrieved: ', len(filtered_df)/len(synergy_df))
    print('selected cell lines: ', filtered_df['cell_line_name'].unique())
    print_synergy_stat(filtered_df)

    return filtered_df
