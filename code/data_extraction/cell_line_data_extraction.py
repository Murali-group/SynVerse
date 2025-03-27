#!/usr/bin/env python
import pandas as pd
import os
from itertools import combinations
from data_extraction.utils import *

def extract_cosmic_data(cosmic_filename):
    cosmic_df = pd.read_csv(cosmic_filename, usecols=['COSMIC_SAMPLE_ID', 'SAMPLE_NAME'], sep='\t')
    cosmic_cell_lines = list(cosmic_df['SAMPLE_NAME'].unique())

    # save cell line names present in CCLE
    cell_lines = sorted(cosmic_cell_lines)
    with open(os.path.dirname(cosmic_filename) + '/cell_lines_in_COSMIC.txt', 'w') as file:
        for cell_line in cell_lines:
            file.write(cell_line + '\n')
    file.close()
    return cosmic_df, cosmic_cell_lines


def map_cclename_to_primary_name(ccle_cell_id_file):
    # now convert the column names from CCLE_cell_line name to primary cell line name
    cell_line_id_df = pd.read_csv(ccle_cell_id_file, sep='\t')[['CCLE name', 'Cell line primary name']]
    name_mappings = dict(zip(cell_line_id_df['CCLE name'], cell_line_id_df['Cell line primary name']))
    return name_mappings

def extract_ccle_data(ccle_expr_file, ccle_cell_id_file, processed_filename, force_run=True):
    if not os.path.exists(processed_filename) or force_run:
        with open(ccle_expr_file) as file:
            line = file.readline()
            col_names = line.split('\t')
        #there are columns containing the presence information i.e., 'P' for present or 'A' for absent, corresponding to each cell line name.
        #But these columns have no name. Name those columns according to the associated cell_line names (which is just the previous col in the file).
        #last column with empty column name is not appearing. so adding it manually.
        col_names.append('')

        # count=0
        for i in range(len(col_names)):
            col_names[i] = col_names[i].replace('\n','') #remove any new line
            if col_names[i]=='':
                # count+=1
                col_names[i] = f'{col_names[i-1]}_presence'
        # there might be duplicate columns (i.e., cell lines)
        duplicates = {x:col_names.count(x) for x in col_names if col_names.count(x) > 1}
        print('duplicate cell lines: ', duplicates)
        print('unique cell lines: ', (len(col_names)-2-len(duplicates))/2)
        #give duplicate cell lines serial number. take care of them later

        for i in range(len(col_names)):
            if col_names[i] in duplicates:
                duplicates[col_names[i]] = duplicates[col_names[i]]-1 #reducing count of repetition
                col_names[i] = f'{col_names[i]}_dup_{duplicates[col_names[i]]}'#add i to the column name for ith repetition

        #read the gene expression file
        ccle_df = pd.read_csv(ccle_expr_file, sep='\t', index_col=False, header=0, names=col_names, skiprows=3, on_bad_lines='warn')

        #take care of the duplicate columns i.e., information of the same cell lines coming multiple times
        #if the presence info matches 90% of the time for the duplicate cell lines then get the average them, otherwise drop both
        ccle_cols = ccle_df.columns
        keep_dup_cols = []
        dup_mean_df = pd.DataFrame()
        for col_name in duplicates:
            if 'presence' in col_name:
                spec_dup_presence_col = [x for x in ccle_cols if f'{col_name}_dup' in x]
                spec_dups_presence_df = ccle_df[spec_dup_presence_col]

                # Create a new DataFrame to store the differences
                diff_df = pd.DataFrame(index=spec_dups_presence_df.index)
                col_pairs = combinations(spec_dups_presence_df.columns, 2)
                # Iterate over all pairs of columns and calculate the differences
                keep=True
                for (col1, col2) in col_pairs:
                    if col1 != col2:
                        # Create a new column name for each pair
                        new_col_name = f'Difference {col1}-{col2}'
                        # Calculate the difference between the pair of columns
                        diff_df[new_col_name] = spec_dups_presence_df[col1] == spec_dups_presence_df[col2]
                        #check if 90% of values are True, i.e., in duplicated columns 90% of
                        # value agree about the presence of a gene in the cell line.
                        true_count = diff_df[new_col_name].sum()
                        prcnt_true = true_count/len(diff_df[new_col_name])

                        if prcnt_true<0.9:
                            keep=False
                            break
                        #     cell_line_name = col_name.replace('presence','')
                        #     remove_dups = [x for x in ccle_cols if cell_line_name in x]
                        #     remove_cols +=remove_dups
                if keep:
                    cell_line_name = col_name.replace('_presence', '')
                    keep_dup_cols.append(cell_line_name)

            else:
                cell_line_name = col_name
                #compute average across duplicate columns. Whether or not to use them will be decided later.
                spec_dup_col = [x for x in ccle_cols if (f'{col_name}_dup' in x and 'presence' not in x)]
                dup_mean_df[cell_line_name] = ccle_df[spec_dup_col].mean(axis=1)


        #now keep the columns with the gene expression values only. Drop the columns with presence/absence info
        # Also drop the duplicate cols at this point.
        keep_columns = [x for x in ccle_df.columns if ('presence' not in x and 'dup' not in x)]
        ccle_df = ccle_df[keep_columns]

        #now add the average gene expression value across duplicated cell lines that appears in keep_dup_cols
        ccle_df=pd.concat([ccle_df, dup_mean_df[keep_dup_cols]], axis=1)

        # remove the 'Accession' column
        ccle_df.drop(columns=['Accession'], inplace=True)
        # remove any row with 'nan' as gene name
        ccle_df.dropna(subset=['Description'], inplace=True)
        #************ CCLE cell line name to primary name mapping
        # now convert the column names from CCLE_cell_line name to primary cell line name
        name_mappings = map_cclename_to_primary_name(ccle_cell_id_file)

        #The columns in ccle_df are now for each cell line. Replace the ccle defined cell line names with primary cell line names.
        #This primary cell line names will be matched with cell line names present in synergy dataset later.
        ccle_df.rename(name_mappings, inplace=True, axis=1)
        # # rename cell line names removing ‘ ’, ‘_’, ‘-’, lowercase
        converted_ccle_cell_line_names = {x: convert_cell_line_name(x) for x in ccle_df.columns}
        ccle_df.rename(converted_ccle_cell_line_names, axis=1, inplace=True)

        #What to do with duplicate genes?
        #take average across duplicated genes
        # compute average across duplicate columns.
        ccle_df = ccle_df.groupby('description').mean()

        # #now transpose the ccle_df such that the rows contains cell_line names and column contains gene names
        ccle_df = ccle_df.T
        #after transpose description=cell_line_name
        #save cell line names present in CCLE
        ccle_df.reset_index(names='cell_line_name', inplace=True)
        cell_lines = set(ccle_df['cell_line_name'])
        cell_lines = sorted(cell_lines)
        with open(os.path.dirname(ccle_expr_file) + '/cell_lines_in_CCLE.txt', 'w') as file:
            for cell_line in cell_lines:
                file.write(cell_line + '\n')
        file.close()
        # save processed CCLE expr file
        os.makedirs(os.path.dirname(processed_filename), exist_ok=True)
        ccle_df.to_csv(processed_filename, index=False, sep='\t')

    ccle_df = pd.read_csv(processed_filename, sep='\t')
    cell_lines = set(ccle_df['cell_line_name'])

    return ccle_df, cell_lines

if __name__=='__main__':
    if 'snakemake' in globals():
        # ccle_expr_file = snakemake.input[0]
        ccle_expr_file = snakemake.input.ccle_expr_file
        ccle_cell_id_file =  snakemake.input.ccle_cell_id_file
        out_file = snakemake.output.processed_ccle_file
    else:
        ccle_expr_file = '/home/grads/tasnina/Projects/SynVerse/datasets/cell-line/CCLE2012/CCLE_Expression_2012-09-29.res'
        ccle_cell_id_file = '/home/grads/tasnina/Projects/SynVerse/datasets/cell-line/CCLE2012/CCLE_Expression.Arrays.sif_2012-10-18.txt'
        out_file = '/home/grads/tasnina/Projects/SynVerse/inputs/cell-line/gene_expression.tsv'

    ccle_df, ccle_cell_lines = extract_ccle_data(ccle_expr_file, ccle_cell_id_file, out_file, force_run=True)
