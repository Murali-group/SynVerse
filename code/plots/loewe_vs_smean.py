import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def plot_violin(data, title, out_file):
    #Plot scores across multiple runs
    # plt.figure(figsize=(6, 4))
    # sns.violinplot(x="run", y="score_value", hue="score_name", data=data, split=False,
    #                palette="muted", linewidth=0.4, flierprops=dict(marker='o', color='gray', markersize=2, markerfacecolor='none', markeredgewidth=0.1))
    sns.boxplot(x="run", y="score_value", hue="score_name", data=data, linewidth=0.4,
                flierprops=dict(marker='o', color='gray', markersize=2,markerfacecolor='none', markeredgewidth=0.2))

    plt.xlabel("Run")
    plt.ylabel("Score Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_file}_loewe_vs_S_dist_box_plot.png')

    plt.show()


    #Plot scores across train and test

    plt.clf()
    plt.figure(figsize=(4, 6))
    sns.violinplot( x ='dataset', y="score_value", hue="score_name",data=data, palette="muted", linewidth=0.4,
               flierprops=dict(marker='o', color='gray', markersize=2,markerfacecolor='none', markeredgewidth=0.2))
    # plt.title(f"Score Distribution for {dataset} ({feature_prefix})")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'{out_file}_loewe_vs_S_dist_violin_plot.pdf')
    plt.show()

def main():
    plot_dir = "/home/grads/tasnina/Projects/SynVerse/inputs/plot/"
    split_dir = "/home/grads/tasnina/Projects/SynVerse/inputs/splits/"
    feature_prefixes = {'D_ECFP_4_MACCS_MFP_d1hot_mol_graph_smiles_C_c1hot':'SMILES', 'D_d1hot_target_C_c1hot':'Target'}
    split_types = ['leave_drug', 'leave_cell_line']
    score_names_map = {'S_mean_mean':'S_mean', 'synergy_loewe_mean':'Loewe'}
    dataset_map = {'all_train':'Train', 'test':'Test'}
    for feature_prefix in feature_prefixes:
        for split_type in split_types:
            combined_data = []
            for dataset in dataset_map:
                for run in range(5):
                    for score_name in score_names_map:
                        file_name = os.path.join(split_dir, feature_prefix, f'k_0.05_{score_name}',
                                                 f'{split_type}_0.2_0.25', f'run_{run}', f'{dataset}.tsv')

                        if os.path.exists(file_name):
                            df = pd.read_csv(file_name, sep="\t")

                            # Add metadata to the dataframe
                            df["run"] = run
                            df["score_name"] = score_names_map[score_name]
                            df["score_value"] = df[score_name]  # Assuming score_name column exists
                            df['dataset'] = dataset_map[dataset]
                            combined_data.append(df[["run", "score_name", "score_value", "dataset"]])

            if combined_data:
                combined_df = pd.concat(combined_data, ignore_index=True)
                out_file = f'{plot_dir}/{split_type}_{feature_prefix}'
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                title = f'{feature_prefixes[feature_prefix]}_{split_type}'
                plot_violin(combined_df,title=title, out_file=out_file)


if __name__ == '__main__':
    main()

