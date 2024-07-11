from synergy_data_prep import *
from drug_target_preprocess import *
from cell_line_preprocess import *
from drug_property_preprocess import *

def data_process():
    #***************************************** SYNERGY *****************************************************************
    #download and process synergy labels
    dcomb_raw_syn_file = '/home/grads/tasnina/Projects/Plug and Play/datasets/synergy/drugcomb_raw_summary.csv'
    dcomb_drug_file = '/home/grads/tasnina/Projects/Plug and Play/datasets/synergy/drugcomb_drug_data.csv'
    dcomb_cell_line_file = '/home/grads/tasnina/Projects/Plug and Play/datasets/synergy/drugcomb_cell_line_data.csv'
    mapped_syn_filename = dcomb_raw_syn_file.replace('raw', 'mapped')

    # download synergy triplets and information on drug and cell lines from drugcomb
    # todo: Uncomment
    download_synergy(dcomb_raw_syn_file, dcomb_drug_file, dcomb_cell_line_file)
    # map drugcomb drug_id->drug_name, cell_line_id->cell_line_name
    mapped_syn_df = map_drugcomb_ids(dcomb_raw_syn_file, dcomb_drug_file, dcomb_cell_line_file, mapped_syn_filename)

    #***************************************
    #**************************************  DRUG **********************************************************************
    #***************************************
    #  extract pubchempy compound for each drug and save that id alongside drug names
    drug_name_to_pcomp_file = '/home/grads/tasnina/Projects/Plug and Play/inputs/drug/drug_name_to_pubchem_compound.pickle'
    synergy_df = pd.read_csv(mapped_syn_filename, sep=',', low_memory=False,
                               dtype={'drug_row': str, 'drug_col': str, 'cell_line_name': str,
                                      'S_mean': float, 'synergy_zip': float})
    drug_names = list(set(synergy_df['drug_row']).union(set(synergy_df['drug_col'])))
    drug_name_to_pcomp = wrapper_download_pubchem_compound(drug_names, drug_name_to_pcomp_file, force_run=False)
    print('drugs with pubchem compounds: ', len(list(drug_name_to_pcomp.keys())))
    #***************************************

    # **************************************
    # aggregate synergy score for replicated triplets and do some filtering.
    processed_syn_file = '/home/grads/tasnina/Projects/Plug and Play/inputs/synergy/synergy_scores.tsv'
    stat_file = '/home/grads/tasnina/Projects/Plug and Play/outputs/stat/synergy_stat.txt'
    synergy_df = aggregate_filter_synergy_data(mapped_syn_filename, drug_name_to_pcomp_file, processed_syn_file, stat_file)  # preprocess
    synergy_df = synergy_df[['drug_1_pid', 'drug_2_pid', 'cell_line_name', 'S_mean_mean', 'synergy_zip_mean']]

    print('Drugcomb: #triplets after aggregation and filtering: ', len(synergy_df))
    # ************************************************************

    #***************************************
    # extract and save drug_smiles
    drug_smiles_file = '/home/grads/tasnina/Projects/Plug and Play/inputs/drug/drug_smiles.tsv'
    drug_smiles_df = extract_pid_smiles(drug_name_to_pcomp_file, drug_smiles_file, force_run=True)
    #**************************************


    # # # filter synergy labels keeping only the drugs having  PID
    drugs_with_smiles = set(drug_smiles_df['pid'].unique())
    synergy_df_smiles = synergy_df[(synergy_df['drug_1_pid'].isin(drugs_with_smiles)) &
                            (synergy_df['drug_2_pid'].isin(drugs_with_smiles))]
    print('#synergy triplets with SMILES information: ', len(synergy_df_smiles))
    print('##unique drugs(pids) with smiles: ', len(set(synergy_df_smiles['drug_1_pid']).
                        union(synergy_df_smiles['drug_2_pid'])))



    #Extract graph from SMILES using deepchem
    drug_graph_file = '/home/grads/tasnina/Projects/Plug and Play/inputs/drug/drug_molecular_graph.pickle'
    pid_to_adjacency_mol_feat = get_graph_from_smiles(drug_smiles_df, drug_graph_file, force_run=True)


    #Extract drug chemical properties using RDKit
    drug_chemprop_file = '/home/grads/tasnina/Projects/Plug and Play/inputs/drug/drug_chemprop/'
    properties = ['MACCS', 'MFP', 'ECFP_4']
    pid_chemprop_dfs = get_chemprop_from_smiles(drug_smiles_df, properties, drug_chemprop_file, force_run=True)

    # #extract target info from TTD
    # drug_target_file = '/home/grads/tasnina/Projects/Plug and Play/inputs/drug/drug_target.tsv'
    TTD_drug_target_file = '/home/grads/tasnina/Projects/Plug and Play/datasets/drug/TTD/P1-07-Drug-TargetMapping.csv'
    TTD_target_file = '/home/grads/tasnina/Projects/Plug and Play/datasets/drug/TTD/P2-01-TTD_uniprot_all_metadata_removed.txt'
    TTD_drug_file = '/home/grads/tasnina/Projects/Plug and Play/datasets/drug/TTD/P1-03-TTD_crossmatching_metadata_removed.txt'
    TTD_drug_target_df = get_drug_targets_from_TTD(TTD_drug_target_file, TTD_target_file,TTD_drug_file, drug_name_to_pcomp_file)

    # filter synergy labels keeping only the drugs having target
    pids_with_target = set(TTD_drug_target_df['pid'].unique())
    synergy_df_target = synergy_df[(synergy_df['drug_1_pid'].isin(pids_with_target)) &
                                   (synergy_df['drug_2_pid'].isin(pids_with_target))]
    print('#of triplets after filtering according to target info: ', len(synergy_df_target))
    print('##unique drugs(pid) with targets: ', len(set(synergy_df_target['drug_1_pid']).
                                                      union(synergy_df_target['drug_2_pid'])))

    # extract target info from Drug Bank
    # db_drug_target_file = "/home/grads/tasnina/Projects/Plug and Play/datasets/drug/Drugbank/drugbank_all_full_database.xml.zip"
    # db_drug_target_df = parse_drugbank_data(db_drug_target_file, drug_name_to_pcomp_file)

    # #******************************** CELL LINE DATA ********************************************************************
    ccle_expr_file = '/home/grads/tasnina/Projects/Plug and Play/datasets/cell-line/CCLE2012/CCLE_Expression_2012-09-29.res'
    ccle_cell_id_file = '/home/grads/tasnina/Projects/Plug and Play/datasets/cell-line/CCLE2012/CCLE_Expression.Arrays.sif_2012-10-18.txt'
    ccle_df, ccle_cell_lines = extract_ccle_data(ccle_expr_file, ccle_cell_id_file, force_run=True)


    ## ***************************************** SUMMARY ****************************************

    # cell line present in synergy data
    drugcomb_cell_lines = set(synergy_df['cell_line_name'])
    ccle_cell_lines = set(ccle_df.columns)
    common_cell_lines = drugcomb_cell_lines.intersection(ccle_cell_lines)

    # filter by keeping only the triplets with cell line information.
    synergy_df_cell_line = synergy_df[synergy_df['cell_line_name'].isin(common_cell_lines)]
    print('#of triplets after filtering according to cell line data: ', len(synergy_df_cell_line))
    print('#of cell lines: ', len(synergy_df_cell_line['cell_line_name'].unique()))
    synergy_df_cell_line.groupby(['cell_line_name']).count()
    print('##unique drugs(pid) with cell line info: ', len(set(synergy_df_cell_line['drug_1_pid']).
                                                    union(synergy_df_cell_line['drug_2_pid'])))


data_process()