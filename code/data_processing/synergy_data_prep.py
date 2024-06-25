#!/usr/bin/env python
from data_utils.helper_functions import *
from drug_property_preprocess import *


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
                'drug_col_id', 'cell_line_id', 'S_mean', 'synergy_zip'], low_memory=False)
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



def aggregate_filter_synergy_data(mapped_filename, drug_name_to_pcomp_file, processed_syn_file, stat_filename, force_run=True):
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
                    'S_mean':float, 'synergy_zip':float})

        drug_names = list(set(synergy_df['drug_row']).union(set(synergy_df['drug_col'])))
        print_stat(synergy_df, drug_names)

        #keep the rows with a pairs of drugs present, NOT single drug respose.
        synergy_df = synergy_df.dropna(subset=['drug_row', 'drug_col'])
        drug_names = list(set(synergy_df['drug_row']).union(set(synergy_df['drug_col'])))
        print_stat(synergy_df, drug_names)

        #check if it works
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
        synergy_df = synergy_df.groupby(['drug_1_pid','drug_2_pid','cell_line_name'])[['S_mean','synergy_zip']].agg(['mean','median', 'std']).reset_index()
        #std=nan means only one sample was present, hence 0 standard deviation
        synergy_df.columns = ['_'.join(col) if col[1]!='' else col[0] for col in synergy_df.columns.values]
        print(synergy_df.columns)
        synergy_df['S_mean_std'] = synergy_df['S_mean_std'].fillna(0)
        synergy_df['synergy_zip_std'] = synergy_df['synergy_zip_std'].fillna(0)

        #Dealing with inconsistent replicates: We only take the pairs for whom standard deviation of synergy score < 0.1
        synergy_df = synergy_df[(synergy_df['S_mean_std']<0.1) & (synergy_df['synergy_zip_std']<0.1) ]
        drug_names = list(set(synergy_df['drug_1_pid']).union(set(synergy_df['drug_2_pid'])))
        # print_stat(synergy_df, drug_names)

        #rename cell line names removing ‘ ’, ‘_’, ‘-’, lowercase
        synergy_df['cell_line_name'] = synergy_df['cell_line_name'].apply(lambda x: convert_cell_line_name(x))

        #************************************ SAVE to file **********************************************
        #save the processed synergy scores
        os.makedirs(os.path.dirname(processed_syn_file), exist_ok=True)
        synergy_df.to_csv(processed_syn_file, sep='\t')
        #save in stat file
        os.makedirs(os.path.dirname(stat_filename), exist_ok=True)
        f = open(stat_filename, 'w')
        f.write(f'number of unique drug-pair-cellline triplets: {len(synergy_df)}')
        f.close()
    else:
        synergy_df = pd.read_csv(processed_syn_file, sep='\t', dtype={'drug_1_pid':str,'drug_2_pid':str,'cell_line_name':str,
                    'S_mean_mean':float, 'S_mean_median':float, 'S_mean_std':float,
                    'synergy_zip_mean':float, 'synergy_zip_median':float, 'synergy_zip_std':float})
    drug_names = list(set(synergy_df['drug_1_pid']).union(set(synergy_df['drug_2_pid'])))
    print_stat(synergy_df, drug_names)

    return synergy_df

def print_stat(synergy_df, drug_names):
    print(f'Drugcomb: #drug-pair-cellline triplets: {len(synergy_df)}')
    print('Drugcomb: #unique drugs with synergy labels: ', len(drug_names))
    print('Drugcomb: #unique cell lines with synergy labels: ', len(set(synergy_df['cell_line_name'])))

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
            aggregate_filter_synergy_data(mapped_syn_filename, drug_name_to_pcomp_file, processed_syn_file, stat_filename)
    else:
        dcomb_raw_syn_file = "/home/grads/tasnina/Projects/SynVerse/datasets/synergy/drugcomb_raw_summary.csv"
        dcomb_drug_file = "/home/grads/tasnina/Projects/SynVerse/datasets/synergy/drugcomb_drug_data.csv"
        dcomb_cell_line_file = "/home/grads/tasnina/Projects/SynVerse/datasets/synergy/drugcomb_cell_line_data.csv"
        mapped_syn_filename = "/home/grads/tasnina/Projects/SynVerse/datasets/synergy/drugcomb_mapped_summary.csv"
        drug_name_to_pcomp_file = '/home/grads/tasnina/Projects/SynVerse/inputs/drug/drug_name_to_pubchem_compound.pickle'
        processed_syn_file = '/home/grads/tasnina/Projects/SynVerse/inputs/synergy/synergy_scores.tsv'
        stat_file = '/home/grads/tasnina/Projects/SynVerse/outputs/stat/synergy_stat.txt'


        # **************************************
        # download_synergy(dcomb_raw_syn_file, dcomb_drug_file, dcomb_cell_line_file)
        # map drugcomb drug_id->drug_name, cell_line_id->cell_line_name
        # mapped_syn_df = map_drugcomb_ids(dcomb_raw_syn_file, dcomb_drug_file, dcomb_cell_line_file,
        #                                         mapped_syn_filename)
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
        synergy_df = aggregate_filter_synergy_data(mapped_syn_filename, drug_name_to_pcomp_file, processed_syn_file, stat_file, force_run=False)  # preprocess
        synergy_df = synergy_df[['drug_1_pid', 'drug_2_pid', 'cell_line_name', 'S_mean_mean', 'synergy_zip_mean']]

        viz_synergy_data(synergy_df)

        print('Drugcomb: #triplets after aggregation and filtering: ', len(synergy_df))

