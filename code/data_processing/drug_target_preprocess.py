#!/usr/bin/env python
from drug_property_preprocess import *
import zipfile
import xml.etree.ElementTree as ET

##************************************************* TTD ****************************
def parse_ttd_target_file(target_file):
    # Initialize empty lists to store data
    target_ids = []
    uniproids = []
    targ_names = []
    targ_types = []

    # Read the file
    with open(target_file, 'r') as file:
        lines = file.readlines()
        lines = [line for line in lines if line != '\n']  # remove empty lines
        total_lines = len(lines)
        cur = 0
        # Iterate through the lines
        while (cur + 3 < total_lines):
            # for line in lines:
            line = lines[cur]
            cur += 1
            parts = line.strip().split('\t')
            if parts[0] == 'TARGETID':
                target_ids.append(parts[1])
            else:
                target_ids.append('-')

            line = lines[cur]
            cur += 1
            parts = line.strip().split('\t')
            if parts[0] == 'UNIPROID':
                uniproids.append(parts[1])
            else:
                uniproids.append('-')

            line = lines[cur]
            cur += 1
            parts = line.strip().split('\t')
            if parts[0] == 'TARGNAME':
                targ_names.append(parts[1])
            else:
                targ_names.append('-')

            line = lines[cur]
            cur += 1
            parts = line.strip().split('\t')
            if parts[0] == 'TARGTYPE':
                targ_types.append(parts[1])
            else:
                targ_types.append('-')

    # Create a dictionary containing the data
    data = {
        'target_id': target_ids,
        'uniprot_id': uniproids,
        'target_name': targ_names,
        'target_type': targ_types
    }

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    # now keep the targets from human
    df = df[df['uniprot_id'].str.contains('HUMAN')]
    # now remove '_HUMAN' from uniprot id
    # df['uniprot_id'] = df['uniprot_id'].apply(lambda x:x.replace('_HUMAN',''))
    # Print the DataFrame
    # print(df.head(5))
    return df


def get_pid_from_ttd(TTD_drug_file, drug_target_df, compound_file):
    # get drug name from ttd_id
    ttd_id_2_drug_name_dict = get_ttd_id_2_drug_name(TTD_drug_file)
    drug_target_df['drug_name'] = drug_target_df['ttd_id'].apply(lambda x: ttd_id_2_drug_name_dict[x]
    if x in ttd_id_2_drug_name_dict else np.nan)
    drug_target_df.dropna(axis=0, inplace=True)  # drop rows if no drugname found for ttd_id

    # get pubchem_id of the drug_names present in drug_target_df
    drug_names = sorted(list(set(drug_target_df['drug_name'].unique())))

    # get pubchem compounds in batches of size 2000
    bin_size = 500
    n_bins = int(len(drug_names) / bin_size) + 1
    pc_compounds = {}
    for i in range(n_bins):
        binwise_compound_file = compound_file.replace('.pickle', f'_bin{i}.pickle')
        if (i + 1) * bin_size < len(drug_names):
            pc_compounds.update(
                wrapper_download_pubchem_compound(drug_names[i * bin_size:(i + 1) * bin_size], binwise_compound_file))
        else:
            pc_compounds.update(wrapper_download_pubchem_compound(drug_names[i * bin_size:], binwise_compound_file))
        # print(f'bin {i} done')

    # filter out drugs having multiple smiles which are unequal.
    mult_uneq_smiles = find_drugs_with_ueq_smiles(pc_compounds)
    for index, row in drug_target_df.iterrows():
        if row['drug_name'] in (pc_compounds.keys()):
            drug_target_df.at[index, 'pid'] = pc_compounds[row['drug_name']][0].cid if pc_compounds[row[
                'drug_name']] not in mult_uneq_smiles else np.nan
    return drug_target_df


def get_ttd_id_2_drug_name(drug_pubchemID_mapping_file):
    df = pd.read_csv(drug_pubchemID_mapping_file, sep='\t', header=None)
    # drugid-drugname mapping df
    drug_name_df = df[df[1]. \
        isin(['DRUGNAME'])][[0, 2]]. \
        rename(columns={0: 'ttd_id', 2: 'drug_name'})
    return dict(zip(drug_name_df['ttd_id'], drug_name_df['drug_name']))


def get_drug_targets_from_TTD(drug_target_file, TTD_target_file, TTD_drug_file, compound_file, out_file=None):
    drug_target_df = pd.read_csv(drug_target_file).rename(columns={'TargetID': 'target_id', 'DrugID': 'ttd_id'})
    drug_target_df = get_pid_from_ttd(TTD_drug_file, drug_target_df, compound_file)

    targetid_uniprot_mapping_df = parse_ttd_target_file(TTD_target_file)
    target_id_to_uniprot_dict = dict(
        zip(targetid_uniprot_mapping_df['target_id'], targetid_uniprot_mapping_df['uniprot_id']))
    target_id_to_name_dict = dict(
        zip(targetid_uniprot_mapping_df['target_id'], targetid_uniprot_mapping_df['target_name']))

    drug_target_df['uniprot_id'] = drug_target_df['target_id'].apply(
        lambda x: target_id_to_uniprot_dict[x] if x in target_id_to_uniprot_dict else np.nan)
    drug_target_df['target_name'] = drug_target_df['target_id'].apply(
        lambda x: target_id_to_name_dict[x] if x in target_id_to_name_dict else np.nan)

    if out_file is not None:
        drug_target_df.to_csv(drug_target_df,sep='\t')
    # drug_target_df.dropna(axis=0, inplace=True)
    # print(drug_target_df.head(10))
    return drug_target_df


if __name__=='__main__':
    TTD_drug_target_file = snakemake.input[0]
    TTD_target_file =  snakemake.input[1]
    TTD_drug_file =  snakemake.input[2]
    drug_name_to_pcomp_file=snakemake.input[3]
    out_file=snakemake.output[0]
    TTD_drug_target_df = get_drug_targets_from_TTD(TTD_drug_target_file, TTD_target_file, TTD_drug_file,
                                                   drug_name_to_pcomp_file, out_file)

