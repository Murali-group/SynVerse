import os
os.environ['R_HOME'] = '/home/tasnina/miniconda3/envs/decagon/lib/R'
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import numpy as np
import pandas as pd


def compute_roc_auprc(output_df):
    prroc = importr('PRROC')
    prCurve = prroc.pr_curve(scores_class0 = FloatVector(list(output_df['predicted'].values)),
              weights_class0 = FloatVector(list(output_df['true'].values)))

    fpr, tpr, thresholds = roc_curve(y_true=output_df['true'],
                                     y_score=output_df['predicted'], pos_label=1)

    prec, recall, thresholds = precision_recall_curve(y_true=output_df['true'],
                                                      probas_pred=output_df['predicted'], pos_label=1)
    return prec, recall, fpr, tpr, prCurve[2][0], auc(fpr, tpr)

def keep_one_from_symmetric_pairs(df, aggregate = 'max'):
    # df will contain predicted score for all (x,y) and (y,x) pairs from a single run
    pairs_1_2 = df[df['drug_1'] > df['drug_2']]
    pairs_2_1 = df[df['drug_1'] < df['drug_2']]


    pairs_1_2 = pairs_1_2.sort_values(['drug_1', 'drug_2', 'cell_line'], ascending=(True, True, True))
    pairs_2_1 = pairs_2_1.sort_values(['drug_2', 'drug_1', 'cell_line'], ascending=(True, True, True))

    pairs_1_2 = pairs_1_2.reset_index()
    pairs_2_1 = pairs_2_1.reset_index()
    print(pairs_1_2.head())
    print(pairs_2_1.head())

    assert len(pairs_1_2) == len(pairs_2_1), 'problem non-symmetric'


    pairs_1_2['predicted_other_way'] = pd.Series(pairs_2_1['predicted'].values)
    print(pairs_1_2.head())
    if(aggregate =='max'):
        pairs_1_2['predicted'] = pairs_1_2[['predicted','predicted_other_way']].max(axis=1)
        # # again negative pair may come up in multiple fold. take the max/avg of them.
        # pairs_1_2 = pairs_1_2.groupby(['drug_1', 'drug_2', 'cell_line', 'true'], as_index=False)['predicted'].count()
        # print(pairs_1_2.sort_values(by=['predicted'],ascending=False))
        # print(len(pairs_1_2), len(pairs_1_2[pairs_1_2['predicted']>1]))
        pairs_1_2 = pairs_1_2.groupby(['drug_1','drug_2','cell_line','true'], as_index=False)['predicted'].max()
    elif(aggregate =='avg'):
        pairs_1_2['predicted'] = pairs_1_2[['predicted','predicted_other_way']].mean(axis=1)

        # again negative pair may come up in multiple fold. take the max/avg of them.
        pairs_1_2 = pairs_1_2.groupby(['drug_1', 'drug_2', 'cell_line','true'], as_index=False)['predicted'].mean()

    # pairs_1_2 = pairs_1_2[['drug_1', 'drug_2', 'cell_line', 'predicted']]




    print(pairs_1_2.head())
    return pairs_1_2


def PRROC_comb(output_df_all_runs, alg, config_map):
    #output_df_all_runs is a list of 5/10 dfs (as the number of cross val runs) containing prediction result from models
    out_dir = config_map['project_dir'] + config_map['output_dir'] + alg + '/' + 'plots/'
    run_ = 0
    AUPRC={}
    AUROC={}
    for output_df in output_df_all_runs:
        # print(output_df.head())
        output_df_unique_pairs =  keep_one_from_symmetric_pairs(output_df, aggregate = 'max')
        precision, recall, FPR, TPR, AUPRC[run_], AUROC[run_] = compute_roc_auprc(output_df_unique_pairs)
        plot_roc_prc(precision, recall, FPR, TPR, alg, 'combined',run_, out_dir )
        run_ += 1
    return AUPRC, AUROC


def PRROC_per_cell_line(output_df_all_runs, alg, config_map):
    out_dir = config_map['project_dir'] + config_map['output_dir'] + alg + '/' + 'plots/'
    cell_lines = output_df_all_runs[0]['cell_line'].unique()
    run_ = 0
    AUPRC_dict = {cell_line:[] for cell_line in cell_lines}
    AUROC_dict = {cell_line:[] for cell_line in cell_lines}

    #plot separate ROC and PR-curve for each cell line
    for output_df in output_df_all_runs:
        output_df_unique_pairs = keep_one_from_symmetric_pairs(output_df, aggregate='max')
        for cell_line in cell_lines:
            cell_line_specific_df = output_df_unique_pairs[output_df_unique_pairs['cell_line'] == cell_line]
            precision, recall, FPR, TPR, AUPRC, AUROC = compute_roc_auprc(cell_line_specific_df)
            plot_roc_prc(precision, recall, FPR, TPR,alg, cell_line,run_, out_dir)
            AUPRC_dict[cell_line].append(AUPRC)
            AUROC_dict[cell_line].append(AUROC)

        run_ += 1
    #plot AUPRC and AUROC values over multiple runs across all cell lines in one boxplot
    plot_auc_auprc_boxplot_all_cell_line_all_runs(AUPRC_dict, AUROC_dict, out_dir)
    return AUPRC_dict, AUROC_dict


def plot_auc_auprc_boxplot_all_cell_line_all_runs(AUPRC_dict, AUROC_dict,out_dir ):

    #plot AUPRC boxplot
    plot_data=[]
    plot_label=[]

    AUPRC_med_dict = {cell_line: statistics.median(AUPRC_dict[cell_line]) for cell_line in AUPRC_dict}

    AUPRC_med_sorted_dict = dict(sorted(AUPRC_med_dict.items(), key=lambda item: item[1], reverse=True))

    for cell_line in AUPRC_med_sorted_dict:
        plot_data.append(AUPRC_dict[cell_line])
        plot_label.append(cell_line)
    plt.boxplot(plot_data,labels=plot_label)

    plt.xticks(rotation='vertical', fontsize=7)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.2)
    plt.ylabel('AUPRC Score')
    plt.xlabel('Cell lines')

    plt.savefig(out_dir+'auprc_across_all_runs.png')
    plt.savefig(out_dir + 'auprc_across_all_runs.pdf')
    plt.show()
    plt.clf()

    # plot AUROC boxplot
    plot_data = []
    plot_label = []
    AUROC_med_dict = {cell_line: statistics.median(AUROC_dict[cell_line]) for cell_line in AUROC_dict}
    AUROC_med_sorted_dict = dict(sorted(AUROC_med_dict.items(), key=lambda item: item[1], reverse=True))
    for cell_line in AUROC_med_sorted_dict:
        plot_data.append(AUROC_dict[cell_line])
        plot_label.append(cell_line)
    plt.boxplot(plot_data, labels=plot_label)

    plt.xticks(rotation='vertical', fontsize=7)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.2)
    plt.ylabel('AUROC Score')
    plt.xlabel('Cell lines')

    plt.savefig(out_dir + 'auroc_across_all_runs.png')
    plt.savefig(out_dir + 'auroc_across_all_runs.pdf')
    plt.show()
    plt.clf()


def plot_roc_prc(precision, recall, FPR, TPR, alg, cell_line,run_,out_dir):
    ## Make PR curves
    # legendList = []


    out_dir = out_dir + 'run'+str(run_)+'/'
    #to_make cell line name compatible for file name replace \/ in the cell line name
    cell_line_file_compat_name = cell_line.replace('/','_')
    cell_line_file_compat_name = cell_line_file_compat_name.replace('\\','_')

    os.makedirs(out_dir, exist_ok=True)

    sns.lineplot(recall, precision, ci=None)
    # legendList.append(key + ' (AUPRC = ' + str("%.2f" % (AUPRC)) + ')')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(cell_line)
    try:
        plt.savefig(out_dir + cell_line_file_compat_name + '_PRCurve.pdf')
        plt.savefig(out_dir + cell_line_file_compat_name + '_PRCurve.png')
    except:
        print('problem in auroc/auprc saving with: ', cell_line_file_compat_name)
    plt.clf()

    ## Make ROC curves

    sns.lineplot(FPR, TPR, ci=None)
    plt.plot([0, 1], [0, 1], linewidth=1.5, color='k', linestyle='--')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(cell_line)
    try:
        plt.savefig(out_dir + cell_line_file_compat_name+ '_ROC.pdf')
        plt.savefig(out_dir + cell_line_file_compat_name+ '_ROC.png')
    except:
        print('problem in auroc/auprc saving with: ', cell_line_file_compat_name)
    plt.clf()


def plot_predicted_score_distribution(df_all_runs,pos_or_neg):
    count=0
    for df in df_all_runs:
        df_unique_pairs = keep_one_from_symmetric_pairs(df, aggregate='max')
        data = list(df_unique_pairs['predicted'])
        plt.hist(data,\
                 weights=np.zeros_like(data) + 1. / len(data), alpha=0.7)

        plt.title('predicted score distribution_'+pos_or_neg + str(count))
        plt.show()
        plt.clf()
        count += 1


def plot_pairwise_value_difference(positive_df_all_runs):
    # for pos_df in positive_df_all_runs:
    pos_df= positive_df_all_runs[0]
    pairs_1_2 = pos_df[pos_df['drug_1'] > pos_df['drug_2']]
    pairs_2_1 = pos_df[pos_df['drug_1'] < pos_df['drug_2']]
    assert len(pairs_1_2) == len(pairs_2_1), 'problem non-symmetric'

    pairs_1_2 = pairs_1_2.sort_values(['drug_1','drug_2','cell_line'], ascending=(True, True, True))
    pairs_2_1 = pairs_2_1.sort_values(['drug_2','drug_1','cell_line'], ascending=(True, True,True))
    assert len(pairs_1_2) == len(pairs_2_1), 'problem non-symmetric'

    # print('pairs_1_2')
    # print(pairs_1_2[['drug_1','drug_2','cell_line']].head())
    # print('pairs_2_1')
    # print(pairs_2_1[['drug_1','drug_2','cell_line']].head())
    #
    # print(pairs_1_2['predicted'].head())
    # print(pairs_2_1['predicted'].head())
    diff_data = np.array(pairs_1_2['predicted'].values)-np.array(pairs_2_1['predicted'].values)
    diff_data = np.absolute(diff_data)

    plt.boxplot(diff_data, labels=['diff'], autorange=True)
    plt.title('difference between scores of (x,y) and (y,x) pairs')
    plt.show()
    plt.clf()

    bin_seq = np.linspace(0,0.05,20)
    # plt.hist(diff_data)
    plt.hist(diff_data,\
             weights=np.zeros_like(diff_data) + 1. / diff_data.size,bins=bin_seq, alpha=0.7)
    plt.xlabel('difference between scores of (x,y) and (y,x)')
    plt.ylabel('fraction of total pairs')
    plt.show()
    plt.clf()
    data_stat = [i for i in diff_data if abs(i) < 0.01]
    print(len(data_stat), len(diff_data))

