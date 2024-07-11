#!/usr/bin/env python
from drug_property_preprocess import *
import zipfile

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
                uniproids.append(parts[1].split(';')[0])#keep the first only incase multiple name present
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
        'gene_name': uniproids,
        'target_full_name': targ_names,
        'target_type': targ_types
    }

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    # now keep the targets from human
    df = df[df['gene_name'].str.contains('HUMAN')]
    # now remove '_HUMAN' from uniprot id
    df['gene_name'] = df['gene_name'].apply(lambda x:x.replace('_HUMAN',''))
    # Print the DataFrame
    # print(df.head(5))
    return df


def get_pid_from_ttd(TTD_drug_file, drug_target_df, compound_file, force_run=False):
    # get drug name from ttd_id
    ttd_id_2_drug_name_dict = get_ttd_id_2_drug_name(TTD_drug_file)
    drug_target_df['drug_name'] = drug_target_df['ttd_id'].apply(lambda x: ttd_id_2_drug_name_dict[x]
                                if x in ttd_id_2_drug_name_dict else np.nan)
    drug_target_df.dropna(axis=0, inplace=True)  # drop rows if no drugname found for ttd_id

    if (os.path.exists(compound_file)):
        with open(compound_file, 'rb') as file:
            pc_compounds = pickle.load(file)
    else:
        pc_compounds = {}


    if force_run:
        # get pubchem_id of the drug_names present in drug_target_df
        drug_names = sorted(list(set(drug_target_df['drug_name'].unique())))
        print(f'total drugs in TTD: {len(drug_names)}')

        # from drug_names, remove the drugs for which I already exytracted pubchem compounds
        drug_names = list(set(drug_names).difference(set(pc_compounds.keys())))
        print(f'Need to find PID for: {len(drug_names)}')

        # # get pubchem compounds in batches of size 1000
        bin_size = 1000
        n_bins = int(len(drug_names) / bin_size) + 1
        for i in range(n_bins):
            if (i + 1) * bin_size < len(drug_names):
                pc_compounds.update(
                    wrapper_download_pubchem_compound(drug_names[i * bin_size:(i + 1) * bin_size], compound_file, force_run=True))
            else:
                pc_compounds.update(wrapper_download_pubchem_compound(drug_names[i * bin_size:], compound_file, force_run=True))
            print(f'bin {i} done')
    # filter out drugs having multiple smiles which are unequal.
    mult_uneq_smiles = find_drugs_with_ueq_smiles(pc_compounds)
    for index, row in drug_target_df.iterrows():
        if row['drug_name'] in (pc_compounds.keys()):
            drug_target_df.at[index, 'pid'] = pc_compounds[row['drug_name']][0].cid if pc_compounds[row[
                'drug_name']] not in mult_uneq_smiles else np.nan
    #drop rows where no pubchem id (pid) is present for the drug
    drug_target_df.dropna(subset=['pid'], axis=0, inplace=True)
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
    drug_target_df = get_pid_from_ttd(TTD_drug_file, drug_target_df, compound_file, force_run=False)

    targetid_uniprot_mapping_df = parse_ttd_target_file(TTD_target_file)
    target_id_to_uniprot_dict = dict(
        zip(targetid_uniprot_mapping_df['target_id'], targetid_uniprot_mapping_df['gene_name']))
    target_id_to_name_dict = dict(
        zip(targetid_uniprot_mapping_df['target_id'], targetid_uniprot_mapping_df['target_full_name']))

    drug_target_df['gene_name'] = drug_target_df['target_id'].apply(
        lambda x: target_id_to_uniprot_dict[x] if x in target_id_to_uniprot_dict else np.nan)
    drug_target_df['target_full_name'] = drug_target_df['target_id'].apply(
        lambda x: target_id_to_name_dict[x] if x in target_id_to_name_dict else np.nan)

    # drop rows where no target/uniprot_id is present for the drug
    drug_target_df.dropna(subset=['gene_name'], axis=0, inplace=True)
    if out_file is not None:
        drug_target_df.to_csv(out_file,sep='\t')
    # drug_target_df.dropna(axis=0, inplace=True)
    # print(drug_target_df.head(10))
    return drug_target_df


if __name__=='__main__':
    if 'snakemake' in globals():
        TTD_drug_target_file = snakemake.input[0]
        TTD_target_file =  snakemake.input[1]
        TTD_drug_file =  snakemake.input[2]
        drug_name_to_pcomp_file=snakemake.input[3]
        out_file=snakemake.output[0]
    else:
        TTD_drug_target_file = '/home/grads/tasnina/Projects/SynVerse/datasets/drug/TTD/P1-07-Drug-TargetMapping.csv'
        TTD_target_file = '/home/grads/tasnina/Projects/SynVerse/datasets/drug/TTD/P2-01-TTD_uniprot_all_metadata_removed.txt'
        TTD_drug_file = '/home/grads/tasnina/Projects/SynVerse/datasets/drug/TTD/P1-03-TTD_crossmatching_metadata_removed.txt'
        drug_name_to_pcomp_file = '/home/grads/tasnina/Projects/SynVerse/inputs/drug/drug_name_to_pubchem_compound.pickle'
        out_file = '/home/grads/tasnina/Projects/SynVerse/inputs/drug/target.tsv'
    TTD_drug_target_df = get_drug_targets_from_TTD(TTD_drug_target_file, TTD_target_file, TTD_drug_file,
                                                   drug_name_to_pcomp_file, out_file)

