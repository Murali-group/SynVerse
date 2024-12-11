#!/usr/bin/env python
import pandas as pd

from data_utils.helper_functions import *
from drug_property_preprocess import *
import matplotlib.pyplot as plt

def download_synergy(syn_filename, drug_filename, cell_line_filename):
    '''
    This function will download synergy data using drugcomb API and save the information in a comma separated .csv file.
    '''
    drugcomb_base_url = 'https://api.drugcomb.org/'
    #
    # #*********************************** Synergy triplets *********************************
    #download synergy information in bins of size=30000
    bin_size = 30000
    count=0
    synergy_df=pd.DataFrame()
    while True:
        offset = count*bin_size+1
        limit = (count+1)*bin_size
        bin_url = f'{drugcomb_base_url}/summary?from={offset}&to={limit}'
        try:
            response = requests.get(bin_url, verify=False)
            if response.status_code == 200:
                js_reponse = response.json()
                if js_reponse:
                    df = pd.DataFrame(js_reponse)
                    # print(time.time()-t1)
                    # print(count)
                else:
                    break
            synergy_df = pd.concat([synergy_df, df], axis=0)
            count+=1
            time.sleep(0.2)
        except requests.exceptions.SSLError as e:
            print(f"SSL Error encountered: {e}")

    synergy_df.to_csv(syn_filename, sep=',')

    # ******************************** DRUGS **************************************
    # download drug info to map drug_id(present summary data) to drug_name
    drug_url = f'{drugcomb_base_url}/drugs'
    try:
        response = requests.get(drug_url, verify=False)
        if response.status_code == 200:
            drug_df = pd.DataFrame(response.json())
    except requests.exceptions.SSLError as e:
        print(f"SSL Error encountered: {e}")

    drug_df.to_csv(drug_filename, sep=',')


    # **************************** Cell line URL ******************************************
    # download cell line info to map cell_line_id(present summary data) to cell_line_name
    cell_line_url = f'{drugcomb_base_url}/cell_lines'
    try:
        response = requests.get(cell_line_url, verify=False)
        if response.status_code == 200:
            cell_line_df = pd.DataFrame(response.json())
    except requests.exceptions.SSLError as e:
        print(f"SSL Error encountered: {e}")
    cell_line_df.to_csv(cell_line_filename, sep=',')


def map_drugcomb_ids(raw_syn_filename, drug_filename, cell_line_filename, mapped_syn_filename):
    drug_df = pd.read_csv(drug_filename, sep=',')
    cell_line_df = pd.read_csv(cell_line_filename, sep=',')

    synergy_df = pd.read_csv(raw_syn_filename, sep=',', usecols =['drug_row_id',
                'drug_col_id', 'cell_line_id', 'S_mean', 'synergy_zip',
                'synergy_loewe', 'synergy_hsa', 'synergy_bliss'], low_memory=False)
    synergy_df.dropna(inplace=True)

    drug_id_to_name = dict(zip(drug_df['id'], drug_df['dname']))
    cell_line_id_to_name = dict(zip(cell_line_df['id'], cell_line_df['name']))

    synergy_df['drug_row'] = synergy_df['drug_row_id'].astype(int).apply(lambda x: drug_id_to_name[x]
                            if x in drug_id_to_name else np.nan)
    synergy_df['drug_col'] = synergy_df['drug_col_id'].astype(int).apply(lambda x: drug_id_to_name[x]
                            if x in drug_id_to_name else np.nan)
    synergy_df['cell_line_name'] = synergy_df['cell_line_id'].astype(int).apply(lambda x: cell_line_id_to_name[x]
                                if x in cell_line_id_to_name else np.nan)

    # save synergy_df in syn_file
    synergy_df.drop(columns=['drug_row_id', 'drug_col_id', 'cell_line_id'], inplace=True)

    synergy_df.to_csv(mapped_syn_filename, sep=',')
    return synergy_df



def aggregate_synergy_data(mapped_filename, drug_name_to_pcomp_file, processed_syn_file, force_run=True):
    '''
    Function: Given the synergy file (syn_filename) from DrugComb, this function will process this file to
        1. keep only drug pairs, no single drug response
        2. keep only consistent drug pairs (std of synergy score(both S_mean, ZIP) <0.1)
        3. compute the mean, median, and std of synergy score.
    Output: It saves the processed data in a file and also return as a dataframe.
    '''
    if not (os.path.exists(processed_syn_file)) or force_run:

        drug_name_2_pid = extract_unambiguouse_drug_pid(drug_name_to_pcomp_file)
        synergy_df = pd.read_csv(mapped_filename, sep=',', low_memory=False,
                    dtype={'drug_row':str,'drug_col':str,'cell_line_name':str,
                    'S_mean':float, 'synergy_loewe':float,
                    'synergy_zip':float,'synergy_hsa':float,
                    'synergy_bliss':float
                    })

        drug_names = list(set(synergy_df['drug_row']).union(set(synergy_df['drug_col'])))
        print_stat(synergy_df, drug_names)

        #keep the rows with a pairs of drugs present, NOT single drug respose.
        synergy_df = synergy_df.dropna(subset=['drug_row', 'drug_col'])
        drug_names = list(set(synergy_df['drug_row']).union(set(synergy_df['drug_col'])))
        print_stat(synergy_df, drug_names)

        synergy_df['drug_1_pid'] = synergy_df['drug_row'].astype(str).apply(
            lambda x: drug_name_2_pid[x] if x in drug_name_2_pid else pd.NA)
        synergy_df['drug_2_pid'] = synergy_df['drug_col'].astype(str).apply(
            lambda x: drug_name_2_pid[x] if x in drug_name_2_pid else pd.NA)

        #removing drugs without pid
        synergy_df = synergy_df.dropna(subset=['drug_1_pid', 'drug_2_pid'])
        drug_names = list(set(synergy_df['drug_1_pid']).union(set(synergy_df['drug_2_pid'])))
        print_stat(synergy_df , drug_names)

        #sort synergy_df such that drug_row>drug_col. This will help to identify repeated pairs present in the dataset.
        sort_paired_cols(synergy_df, 'drug_1_pid', 'drug_2_pid', inplace=True, relation='greater')
        #Group the same drug-pair-cell-line triplets and take the mean and std scores.
        synergy_df = synergy_df.groupby(['drug_1_pid','drug_2_pid','cell_line_name'])[['S_mean','synergy_zip', 'synergy_loewe', 'synergy_bliss', 'synergy_hsa']].agg(['mean','median', 'std', 'count']).reset_index()
        #TODO: Make sure we are getting nan because of singular presence of a triplet
        #std=nan means only one sample was present, hence 0 standard deviation
        # synergy_df.columns = ['_'.join(col) if col[1]!='' else col[0] for col in synergy_df.columns.values]
        synergy_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in synergy_df.columns]


        #Check *************************
        # Identify columns for std and count
        columns_to_check = [col.split('_std')[0] for col in synergy_df.columns if col.endswith('_std')]

        # Check if std is NaN when count is 1
        for col_prefix in columns_to_check:
            std_col = f"{col_prefix}_std"
            count_col = f"{col_prefix}_count"

            # Filter rows where count is not 1 but  std is NaN (to check for mismatches)
            mismatched_rows = synergy_df[(synergy_df[count_col] != 1) & (synergy_df[std_col].isna())]

            if mismatched_rows.empty:
                print(f"All NaNs in {std_col} are correctly due to {count_col} being 1.")
            else:
                print(f"Discrepancy found in {std_col} when {count_col} is 1.")
                print(mismatched_rows)


        #******************

        print(synergy_df.columns)
        synergy_df['S_mean_std'] = synergy_df['S_mean_std'].fillna(0)
        synergy_df['synergy_zip_std'] = synergy_df['synergy_zip_std'].fillna(0)
        synergy_df['synergy_loewe_std'] = synergy_df['synergy_loewe_std'].fillna(0)
        synergy_df['synergy_hsa_std'] = synergy_df['synergy_hsa_std'].fillna(0)
        synergy_df['synergy_bliss_std'] = synergy_df['synergy_bliss_std'].fillna(0)


        #Dealing with inconsistent replicates: We only take the pairs for whom standard deviation of synergy score < 0.1
        # synergy_df = synergy_df[(synergy_df['synergy_loewe_std']==0)]
        # print_stat(synergy_df, drug_names)

        #rename cell line names removing ‘ ’, ‘_’, ‘-’, lowercase
        synergy_df['cell_line_name'] = synergy_df['cell_line_name'].apply(lambda x: convert_cell_line_name(x))

        #************************************ SAVE to file **********************************************
        #save the processed synergy scores
        os.makedirs(os.path.dirname(processed_syn_file), exist_ok=True)
        synergy_df.to_csv(processed_syn_file, sep='\t')

    else:
        synergy_df = pd.read_csv(processed_syn_file, sep='\t', dtype={'drug_1_pid':str,'drug_2_pid':str,'cell_line_name':str,
                    'S_mean_mean':float, 'S_mean_median':float, 'S_mean_std':float,
                    'synergy_loewe_mean':float, 'synergy_loewe_median':float, 'synergy_loewe_std':float,
                    'synergy_zip_mean': float, 'synergy_zip_median': float,'synergy_zip_std': float,
                    'synergy_hsa_mean': float, 'synergy_hsa_median': float,'synergy_hsa_std': float,
                    'synergy_bliss_mean': float, 'synergy_bliss_median': float,'synergy_bliss_std': float})
    drug_names = list(set(synergy_df['drug_1_pid']).union(set(synergy_df['drug_2_pid'])))
    print('After Aggregation')
    print_stat(synergy_df, drug_names)

    return synergy_df

def print_stat(synergy_df, drug_names):
    unique_drugs = len(drug_names)
    triplets = len(synergy_df)
    cell_lines = len(set(synergy_df['cell_line_name']))
    print(f'Drugcomb: #drug-pair-cellline triplets: {triplets}')
    print('Drugcomb: #unique drugs with synergy labels: ', unique_drugs)
    print('Drugcomb: #unique cell lines with synergy labels: ', cell_lines)
    return unique_drugs, cell_lines, triplets

def viz_synergy_data(synergy_df, out_file=None):
    #distribution of drugs
    # Task 1: Count rows for each cell line
    import matplotlib.pyplot as plt

    cell_line_counts = synergy_df['cell_line_name'].value_counts()[0:109]

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    cell_line_counts.plot(kind='bar')
    plt.title('Distribution of Number of Rows Across Cell Lines')
    plt.xlabel('Cell Line Name')
    plt.ylabel('Number of Rows')
    plt.xticks(rotation=45)
    plt.show()

    # Correct the binning approach for the second plot

    # Define the bins for the ranges
    bins = [0, 100, 200, 500, 1000, 5000, 10000, 20000, 30000, 40000]

    # Create a new column to bin the counts
    cell_line_counts = synergy_df['cell_line_name'].value_counts()
    binned_counts = pd.cut(cell_line_counts, bins=bins)

    # Count the number of cell lines in each bin
    cell_line_range_distribution = binned_counts.value_counts().sort_index()

    # Plot the distribution of cell lines across the defined ranges
    plt.figure(figsize=(12, 8))
    cell_line_range_distribution.plot(kind='bar', color='salmon')
    plt.title('Number of Cell Lines in Different Row Count Ranges')
    plt.xlabel('Number of Rows Range')
    plt.ylabel('Number of Cell Lines')
    plt.xticks(rotation=45)
    plt.show()
    print('done')


def plot_distribution_of_scores(synergy_df, score_name,  out_dir, filter=None):
    """
    Plot the distribution of standard deviation values for a specified score to identify outliers.

    Parameters:
        synergy_df (pd.DataFrame): The input DataFrame containing the score and its standard deviation.
        score_name (str): The base name of the score (e.g., 'synergy_loewe').
    """
    # Construct the column name for the standard deviation
    os.makedirs(stat_dir, exist_ok=True)


    # Check if the column exists in the DataFrame
    if score_name not in synergy_df.columns:
        raise ValueError(f"Column '{score_name}' not found in the DataFrame.")

    # Extract the data for the histogram
    std_values = synergy_df[score_name].dropna()  # Exclude NaN values

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(std_values, bins=30, alpha=0.7, edgecolor='black', log=True)
    plt.title(f"Distribution {score_name}", fontsize=16)
    plt.xlabel(f"{score_name}", fontsize=14)
    plt.ylabel("Frequency (Log Scale)", fontsize=14)
    plt.grid(axis='y', linestyle='-', alpha=0.7)

    # Highlight potential outliers
    # Define an outlier threshold, e.g., values above the 95th percentile
    if filter=='percentile':
        for (percentile, clr) in [(90, 'red'), (95, 'green'), (99, 'blue')]:
            outlier_threshold = np.percentile(std_values, percentile)
            plt.axvline(outlier_threshold, color=clr, linestyle='--',
                        label=f'{percentile}th Percentile (Outlier Threshold: {outlier_threshold:.2f})')

    elif filter=='threshold':
        for (outlier_threshold, clr) in [(0.1, 'red'), (10, 'green'), (15, 'blue')]:
            percentile_rank = find_percentile(std_values, outlier_threshold)
            plt.axvline(outlier_threshold, color=clr, linestyle='--',
                        label=f'For Outlier Threshold: {outlier_threshold:.2f}, Percentile: {percentile_rank:.2f}%')

    plt.legend(fontsize=12)
    plt.savefig(os.path.join(out_dir, f'{score_name}_distribution_{filter}.pdf'))

    plt.show()


def find_percentile(values, given_value):
    """
    Find the percentile of a given value in a list or array of values.

    Parameters:
        values (list or np.ndarray): The array of values.
        given_value (float): The value to find the percentile for.

    Returns:
        float: The percentile rank of the given value.
    """
    # Convert values to a NumPy array
    values = np.array(values)

    # Calculate the percentile rank
    percentile = (np.sum(values < given_value)) / len(values) * 100
    return percentile

if __name__=='__main__':
    #*************************** DETAILED SNAKEMAKE *******************************
    if 'snakemake' in globals():
        if snakemake.params[0] == 'download':
            print('from snakemake')
            download_filename = snakemake.output[0]
            drug_filename = snakemake.output[1]
            cell_line_filename=snakemake.output[2]
            download_synergy(download_filename, drug_filename, cell_line_filename)
        elif snakemake.params[0] == 'map':
            dcomb_raw_syn_file = snakemake.input[0]
            dcomb_drug_file=snakemake.input[1]
            dcomb_cell_line_file=snakemake.input[2]
            mapped_syn_filename=snakemake.output[0]
            map_drugcomb_ids(dcomb_raw_syn_file, dcomb_drug_file, dcomb_cell_line_file, mapped_syn_filename)
        elif snakemake.params[0] == 'aggr':
            mapped_syn_filename = snakemake.input[0]
            drug_name_to_pcomp_file=snakemake.input[1]
            processed_syn_file=snakemake.output[0]
            stat_filename=snakemake.output[1]
            aggregate_synergy_data(mapped_syn_filename, drug_name_to_pcomp_file, processed_syn_file)
    else:
        dcomb_raw_syn_file = "/home/grads/tasnina/Projects/SynVerse/datasets/synergy/drugcomb_raw_summary.csv"
        dcomb_drug_file = "/home/grads/tasnina/Projects/SynVerse/datasets/synergy/drugcomb_drug_data.csv"
        dcomb_cell_line_file = "/home/grads/tasnina/Projects/SynVerse/datasets/synergy/drugcomb_cell_line_data.csv"
        mapped_syn_filename = "/home/grads/tasnina/Projects/SynVerse/datasets/synergy/drugcomb_mapped_summary.csv"
        drug_name_to_pcomp_file = '/home/grads/tasnina/Projects/SynVerse/inputs/drug/drug_name_to_pubchem_compound.pickle'
        processed_syn_file = '/home/grads/tasnina/Projects/SynVerse/inputs/synergy/synergy_scores.tsv'
        stat_dir = '/home/grads/tasnina/Projects/SynVerse/inputs/stat/'
        stat_file = f'{stat_dir}/synergy_stat.tsv'


        # **************************************
        # download_synergy(dcomb_raw_syn_file, dcomb_drug_file, dcomb_cell_line_file)
        # map drugcomb drug_id->drug_name, cell_line_id->cell_line_name
        mapped_syn_df = map_drugcomb_ids(dcomb_raw_syn_file, dcomb_drug_file, dcomb_cell_line_file,
                                                mapped_syn_filename)
        #
        # # #  extract pubchempy compound for each drug and save that id alongside drug names
        # drug_name_to_pcomp_file = '/home/grads/tasnina/Projects/SynVerse/inputs/drug/drug_name_to_pubchem_compound.pickle'
        # synergy_df = pd.read_csv(mapped_syn_filename, sep=',', low_memory=False,
        #                          dtype={'drug_row': str, 'drug_col': str, 'cell_line_name': str,
        #                                 'S_mean': float, 'synergy_zip': float})
        #
        # drug_names = list(set(synergy_df['drug_row']).union(set(synergy_df['drug_col'])))
        # drug_name_to_pcomp = wrapper_download_pubchem_compound(drug_names, drug_name_to_pcomp_file, force_run=False)
        # print('drugs with pubchem compounds: ', len(list(drug_name_to_pcomp.keys())))

        # aggregate synergy score for replicated triplets and do some filtering.
        synergy_df = aggregate_synergy_data(mapped_syn_filename, drug_name_to_pcomp_file, processed_syn_file, force_run=True)  # preprocess
        # synergy_df = synergy_df[['drug_1_pid', 'drug_2_pid', 'cell_line_name', 'S_mean_mean','S_mean_std', 'synergy_loewe_mean', 'synergy_loewe_std']]

        for score_name in ['synergy_loewe', 'S_mean']:
            for filter_option in ['percentile', 'threshold']:
                plot_distribution_of_scores(synergy_df, f'{score_name}_std',stat_dir, filter_option)
            plot_distribution_of_scores(synergy_df, f'{score_name}_mean',stat_dir)
            plot_distribution_of_scores(synergy_df, f'{score_name}_median',stat_dir)


        summar_rows = []
        #filter based on percentile
        for percentile in [90, 95, 99]:
            for score_name in ['synergy_loewe_std', 'S_mean_std']:
                filtered_synergy_file = processed_syn_file.replace('_scores.tsv', f'_{score_name}_percentile_{percentile}.tsv')
                # Calculate the threshold value for the specified percentile
                threshold_value = np.percentile(synergy_df[score_name], percentile)
                # Filter rows where the column value is less than or equal to the threshold
                filtered_df = synergy_df[synergy_df[score_name] <= threshold_value]
                filtered_df.to_csv(filtered_synergy_file, sep='\t')

                print(f'\n\nAfter filtering based on percentile {percentile} where threshold is {threshold_value} on {score_name} ')
                drug_names = list(set(filtered_df['drug_1_pid']).union(set(filtered_df['drug_2_pid'])))
                n_unique_drugs, n_cell_lines, n_triplets = print_stat(filtered_df, drug_names)

                summar_rows.append({'score_name': score_name, 'percentile':percentile, 'threshold':threshold_value,
                        'unique_drugs':n_unique_drugs, 'unique_cell_lines':n_cell_lines, '#triplets':n_triplets})

        #filter based on threshold
        for threshold in [0.1, 10, 15]:
            for score_name in ['synergy_loewe_std', 'S_mean_std']:
                filtered_synergy_file = processed_syn_file.replace('_scores.tsv', f'_{score_name}_threshold_{threshold}.tsv')
                # Filter rows where the column value is greater than or equal to the threshold
                filtered_df = synergy_df[synergy_df[score_name] <= threshold]
                filtered_df.to_csv(filtered_synergy_file, sep='\t')

                #percentile for this threshold
                percentile = find_percentile(synergy_df[score_name], threshold)

                print(f'\n\nAfter filtering based on threshold {threshold} where percentile is {percentile} on {score_name} ')
                drug_names = list(set(filtered_df['drug_1_pid']).union(set(filtered_df['drug_2_pid'])))
                n_unique_drugs, n_cell_lines, n_triplets = print_stat(filtered_df, drug_names)

                summar_rows.append({'score_name': score_name, 'percentile': percentile, 'threshold': threshold,
                                    'unique_drugs': n_unique_drugs, 'unique_cell_lines': n_cell_lines,
                                    '#triplets': n_triplets})

        summary_df = pd.DataFrame(summar_rows)
        summary_df.to_csv(stat_file, sep='\t')


