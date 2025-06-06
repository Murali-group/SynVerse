import re
from sklearn.metrics import recall_score, precision_score, f1_score
import scipy.stats as stats
from .plot_utils import *


def feature_to_filter_map(drug_feat, cell_feat):
    '''
    Based on the features used in a model, map the features to feature_based filter being used in data preprocessing.

    We had a feature-based filtering when we preprocessed the data. Based on what
    feature is being used we have three filters.Here, we map each feature to the filter
    being used for it.
    If the feature has a substring that match with a substring in the filters
    after splitting each of based on '_', we map the feature to that filter. However,
    here feature value may contain feature name, preprocess and encoder info,  e.g.,
    smiles_Transformer, smiles_kpgt both will be mapped to filter D_ECFP_4_MACCS_MFP_d1hot_mol_graph_smiles_C_c1hot.
    '''
    feature_filters = ['D_d1hot_target_C_c1hot', 'D_d1hot_C_c1hot_genex_genex_lincs_1000',
                       'D_ECFP_4_MACCS_MFP_d1hot_mol_graph_smiles_C_c1hot']

    # currently I have run d1hot with AE and c1hot with AE only for SMILES based split.
    if drug_feat in ['d1hot_std_comp_True', 'd1hot_comp_True'] and cell_feat in ['c1hot_std_comp_True', 'c1hot_comp_True']:
        return 'D_ECFP_4_MACCS_MFP_d1hot_mol_graph_smiles_C_c1hot'

    substr_dfeat = drug_feat.split('_')
    substr_cfeat = cell_feat.split('_')


    for feature_filter in feature_filters:
        substr_filter = feature_filter.split('_')
        #both drug and cell feature has to be present
        if set(substr_dfeat).intersection(set(substr_filter)) and (set(substr_cfeat).intersection(set(substr_filter))):

            return feature_filter

def plot_outputs(file_path, split_type):

    # Load the dataset
    df = pd.read_csv(file_path, sep='\t')
    # Grouping columns and metrics
    group_columns = ['drug_features', 'cell_features', 'feature_filter']
    metrics = {
        'Training Loss': 'train_loss',
        'Test Loss': 'test_loss',
        'Precision_0': 'Precision_0',
        'Recall_0': 'Recall_0'
    }
    # Generate plots for each metric
    for metric_name, metric_col in metrics.items():
        grouped = df.groupby(group_columns)[metric_col].agg(['mean', 'std']).reset_index()

        # Create the bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x_labels = grouped[group_columns].agg('-'.join, axis=1)
        x = np.arange(len(x_labels))
        ax.bar(x, grouped['mean'], yerr=grouped['std'], capsize=5)

        # Customize the plot
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{split_type}:{metric_name} Across Groups')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Show the plot
        plt.show()


def compute_cls_performance(pred_file_path, thresholds = [0]):
    #columns = drug1	drug2	cell_line	TRUE	predicted
    pred_df = pd.read_csv(pred_file_path, sep='\t')
    prec_dict = {}
    rec_dict = {}
    f1_dict = {}
    for threshold in thresholds:
        y_true = list(pred_df['true'].astype(float).apply(lambda x : 1 if x > threshold else 0))
        y_pred = list(pred_df['predicted'].astype(float).apply(lambda x : 1 if x > threshold else 0))
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary')
        prec_dict[f'Precision_{threshold}'] = precision
        rec_dict[f'Recall_{threshold}'] = recall
        f1_dict[f'F1_{threshold}'] = f1

    return prec_dict, rec_dict, f1_dict

def compute_corr(pred_file_path):
    #columns = drug1	drug2	cell_line	TRUE	predicted
    pred_df = pd.read_csv(pred_file_path, sep='\t')

    # # Check for NaNs or infs and report rows
    # bad_rows = pred_df[pred_df[['true', 'predicted']].isnull().any(axis=1) |
    #                    ~np.isfinite(pred_df[['true', 'predicted']]).all(axis=1)]
    #
    # if not bad_rows.empty:
    #     print("Warning: Found NaN or Inf values in the following rows:")
    #     print(bad_rows)
    #     return None


    y_true = list(pred_df['true'].astype(float))
    y_pred = list(pred_df['predicted'].astype(float))
    corr_prsn, pval_prsn = stats.pearsonr(y_true, y_pred)
    corr_sprmn, pval_sprmn = stats.spearmanr(y_true, y_pred)
    corr = {'Pearsons':corr_prsn, 'Pearsons_pval': pval_prsn, 'Spearman':corr_sprmn,
            'Spearman_pval': pval_sprmn}
    return corr

def read_loss_file_content(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

        # Extract required information using regular expressions
        best_config_match = re.search(r"Config: (.+)", content)
        epochs_match = re.search(r"Number of epochs: (\d+)", content)
        train_loss_match = re.search(r"train_loss: ([\d.]+)", content)
        test_loss_match = re.search(r"test_loss: ([\d.]+)", content)
        val_loss_match = re.search(r"val_loss: ([\d.]+)", content)


        # Get values if matches are found, else set to None
        best_config = eval(best_config_match.group(1)) if best_config_match else None
        epochs = int(epochs_match.group(1)) if epochs_match else None
        train_loss = float(train_loss_match.group(1)) if train_loss_match else None
        test_loss = float(test_loss_match.group(1)) if test_loss_match else None
        val_loss = float(val_loss_match.group(1)) if val_loss_match else None


        loss_dict = {'test_MSE': test_loss,
            'val_MSE': val_loss,
            'train_MSE': train_loss,
            'num_epochs': epochs,
            'best_config': best_config}
    return loss_dict

def get_run_feat_info(file_path, run_number, feature_filter=None):
    clean_file_name = file_path.split('/')[-1].replace("_val_true_loss.txt", "")
    features = re.sub(r'run_[0-4]', '', clean_file_name)  # Remove 'run_x' pattern
    drug_features = features.split('_C_')[0].replace('D_', '')
    cell_features = features.split('_C_')[1].split('_rewired_')[0].split('_shuffled_')[0].split('_randomized_score_')[0]
    rewired = True if len(features.split('_rewired_'))>1 else False
    if rewired:
        rewire_method = '_'.join(features.split('_rewired_')[-1].split('_')[1:])
    else:
        rewire_method='Original'

    shuffled = True if len(features.split('_shuffled_'))>1 else False
    randomized_score = True if len(features.split('_randomized_score_'))>1 else False

    if shuffled:
        shuffle_method = 'Shuffled'
    else:
        shuffle_method='Original'

    if randomized_score:
        randomized_method = 'Randomized'
    else:
        randomized_method='Original'

    if feature_filter is None:
        feature_filter = feature_to_filter_map(drug_features, cell_features)
    run_info = {
        'run_no': run_number,
        'drug_features': drug_features,
        'cell_features': cell_features,
        'feature_filter': feature_filter,
        'rewired': rewired,
        'rewire_method': rewire_method,
        'shuffled': shuffled,
        'shuffle_method': shuffle_method,
        'randomized_score': randomized_score,
        'randomized_method': randomized_method
    }
    return run_info

def iterate_output_files(folder_path):
    out_file_list = []
    # Iterate over each 'run_x' folder
    for run_folder in os.listdir(folder_path):
        run_path = os.path.join(folder_path, run_folder)

        # Ensure it's a directory and follows the 'run_x' pattern
        if os.path.isdir(run_path) and re.match(r'run_\d+', run_folder):
            run_number = run_folder  # Save run_x as run number
            # Iterate over each file in the 'run_x' folder
            for file_or_dirname in os.listdir(run_path):
                #************* REGRESSION LOSS *********************************
                # Consider files ending with '_loss.txt' to get train and test loss
                if file_or_dirname.endswith('_val_true_loss.txt'):
                    # Open and read the file content
                    loss_file_path = os.path.join(run_path, file_or_dirname)
                    pred_file_path = loss_file_path.replace('_val_true_loss.txt', '_val_true_test_predicted_scores.tsv')
                    #find the feature-based filter it was run on
                    run_info_dict = get_run_feat_info(loss_file_path, run_number)
                    run_info_dict.update({'loss_file':loss_file_path, "pred_file": pred_file_path})
                    out_file_list.append(run_info_dict)

                # read files for one-hot encoding
                if 'One-hot-versions' in file_or_dirname:
                    sub_dirs = os.listdir(os.path.join(run_path, file_or_dirname))
                    for sub_dir in sub_dirs:
                        one_hot_files = os.listdir(os.path.join(run_path, file_or_dirname, sub_dir))
                        for one_hot_file in one_hot_files:
                            if one_hot_file.endswith('_val_true_loss.txt'):
                                # Open and read the file content
                                loss_file_path= os.path.join(run_path, file_or_dirname, sub_dir, one_hot_file)
                                pred_file_path = loss_file_path.replace('_val_true_loss.txt', '_val_true_test_predicted_scores.tsv')
                                run_info_dict = get_run_feat_info(loss_file_path, run_number, feature_filter=sub_dir)
                                run_info_dict.update({'loss_file': loss_file_path, "pred_file": pred_file_path})
                                out_file_list.append(run_info_dict)
    # Create a DataFrame from the collected data
    return out_file_list


def parse_output_files(base_folder):
    # Example usage
    # base_folder=sys.argv[1]
    # base_folder= "/home/grads/tasnina/Projects/SynVerse/outputs/k_0.05_S_mean_mean/"
    # base_folder= "/home/grads/tasnina/Projects/SynVerse/outputs/k_0.05_synergy_loewe_mean/"

    # base_folder= "/home/grads/tasnina/Projects/SynVerse/outputs/sample_norm_0.99/k_0.05_S_mean_mean/"
    # base_folder= "/home/grads/tasnina/Projects/SynVerse/outputs/sample_norm_0.95/k_0.05_S_mean_mean/"
    # base_folder= "/home/grads/tasnina/Projects/SynVerse/outputs/MARSY_data/k_0.05_S_mean_mean/"
    # base_folder= "/home/grads/tasnina/Projects/SynVerse/outputs/MatchMaker_data/k_0.05_synergy_loewe_mean/"
    # base_folder= "/home/grads/tasnina/Projects/SynVerse/outputs/SynergyX_data/k_0.05_S_mean_mean/"

    split_types = ['leave_comb','random', 'leave_drug', 'leave_cell_line']
    outfile_detailed = base_folder + f'combined_output.xlsx'

    #extract output for features other than where both drug and cell lines have one-hot encoding
    #save outputs for each runs across all splits and features
    splitwise_df_dict = {}
    for split_type in split_types:
        splitwise_summary_file = base_folder+f'output_{split_type}'
        spec_folder = f'{base_folder}/{split_type}/'
        if not os.path.exists(spec_folder):
            continue
        out_info_list = iterate_output_files(spec_folder)
        data = []
        for out_info in out_info_list:
            if not os.path.exists(out_info['pred_file']):
                continue
            print(out_info['pred_file'])
            all_info = out_info
            loss_info = read_loss_file_content(out_info['loss_file'])
            correlations_dict  = compute_corr(out_info['pred_file'])
            all_info.update(loss_info)
            all_info.update(correlations_dict)
            data.append(all_info)
        df = pd.DataFrame(data)
        df.drop(columns=['loss_file', 'pred_file'], axis=1, inplace=True)
        splitwise_df_dict[split_type] = df
        print(df.head(5))

        # remove model 'One hot (AE)'
        df = set_model_names(df)
        #I have some extra runs which I don't want to appear in summary files. so remove them
        df = df[df['Model'] != 'One hot (AE)']
        # remove a few feature combo if present. Following remove model where auto-encoder used on one-hot without standardization.
        df = df[~((df['drug_features'] == 'd1hot_comp_True') | (
                df['cell_features'] == 'c1hot_comp_True'))]

        df.dropna(subset=['Model'], inplace=True)

        #seperate performance of models trained on original and rewired training  networks
        df_orig = df[(df['rewired'] == False) & (df['shuffled'] == False) & (df['randomized_score'] == False) ]

        df_rewired = df[df['rewired'] == True]
        df_rewired =df_rewired[df_rewired['Model']!='One hot'] #do not consider one-hot models

        df_shuffled = df[df['shuffled'] == True]
        df_shuffled = df_shuffled[df_shuffled['Model']!='One hot']

        df_randomized = df[df['randomized_score'] == True]
        df_randomized = df_randomized[df_randomized['Model']!='One hot']


        df_orig.to_csv(f'{splitwise_summary_file}.tsv', sep='\t', index=False)
        if not df_rewired.empty:
            df_rewired.to_csv(f'{splitwise_summary_file}_rewired.tsv', sep='\t', index=False)
        if not df_shuffled.empty:
            df_shuffled.to_csv(f'{splitwise_summary_file}_shuffled.tsv', sep='\t', index=False)
        if not df_randomized.empty:
            df_randomized.to_csv(f'{splitwise_summary_file}_randomized.tsv', sep='\t', index=False)

    with pd.ExcelWriter(outfile_detailed, mode="w") as writer:
        for split_type in splitwise_df_dict:
            splitwise_df_dict[split_type].to_excel(writer, sheet_name=split_type, index=False)
