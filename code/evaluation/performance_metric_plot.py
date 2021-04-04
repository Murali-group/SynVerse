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
    # df =  df_input
    df = sort_max_drug_first(df, 'drug_1','drug_2')
    # print('df after sorting\n', df.head())
    if (aggregate == 'max'):
        df = df.groupby(['drug_1','drug_2','cell_line','true'], as_index = False)['predicted'].max()
        # print('df after taking max:\n', df.head())
    elif(aggregate=='mean'):
        df =  df.groupby(['drug_1', 'drug_2', 'cell_line', 'true'], as_index=False)['predicted'].mean()
    return df


def sort_max_drug_first(df, drug1_col, drug2_col):
    #this function will take a dataframe df_input as input
    #sort df_input such that in sorted df, max(drug1_col, drug2_col) will be in drug1_col and min(drug1_col, drug2_col) will be in drug2_col
    # df = df_input
    df['max_drug'] = df[[drug1_col, drug2_col]].max(axis=1)
    df['min_drug'] = df[[drug1_col, drug2_col]].min(axis=1)
    # print(df.head(10))
    df[drug1_col] = df['max_drug']
    df[drug2_col] = df['min_drug']
    df.drop(['max_drug', 'min_drug'], axis=1, inplace=True)
    # print(df.head(10))
    return df


def plot_predicted_score_distribution(pos_df, neg_df, title_suffix, plot_dir):
    # input is from a single run for a single algorithm. this is the minimum unit for plotting

    pos_df_unique_pairs = keep_one_from_symmetric_pairs(pos_df.copy(), aggregate='max')
    neg_df_unique_pairs = keep_one_from_symmetric_pairs(neg_df.copy(), aggregate='max')
    pos_data = list(pos_df_unique_pairs['predicted'])
    neg_data = list(neg_df_unique_pairs['predicted'])
    print(len(pos_data), len(neg_data))
    plt.hist(pos_data, weights=np.zeros_like(pos_data) + 1. / len(pos_data), \
             alpha=0.5, bins=30, color='blue', range=(0, 1), label='positive')
    plt.hist(neg_data, weights=np.zeros_like(neg_data) + 1. / len(neg_data), \
             alpha=0.5, bins=30, color='orange', range=(0, 1), label='negative')

    plt.xlabel('predicted score')
    plt.ylabel('fraction of total (pos or neg) pairs')
    plot_title = 'score distribution ' + title_suffix
    plt.title(plot_title, loc = 'center', wrap=True)
    plt.legend()

    plot_filename_png = plot_dir + plot_title + '.png'
    plot_filename_pdf = plot_dir + plot_title + '.pdf'

    os.makedirs(os.path.dirname(plot_filename_pdf), exist_ok=True)
    plt.savefig(plot_filename_png)
    plt.savefig(plot_filename_pdf)

    plt.show()
    plt.clf()


def plot_auprc_auroc(output_df, title_suffix, plot_dir):
    #scatter plot for auprc and auroc score across all cell line for a certain algorithm at certain run
    cell_lines = output_df['cell_line'].unique()
    output_df_unique_pairs = keep_one_from_symmetric_pairs(output_df.copy(), aggregate='max')

    precision, recall, FPR, TPR, AUPRC, AUROC = {}, {}, {}, {}, {}, {}

    for cell_line in cell_lines:
        cell_line_specific_df = output_df_unique_pairs[output_df_unique_pairs['cell_line'] == cell_line]
        precision[cell_line], recall[cell_line], FPR[cell_line], TPR[cell_line], AUPRC[cell_line], AUROC[cell_line] \
            = compute_roc_auprc(cell_line_specific_df)

    AUPRC_sorted_dict = dict(sorted(AUPRC.items(), key=lambda item: item[1], reverse=True))
    AUROC_sorted_dict = dict(sorted(AUROC.items(), key=lambda item: item[1], reverse=True))

    #plot auprc
    plt.scatter(AUPRC_sorted_dict.keys(), AUPRC_sorted_dict.values())
    plt.xticks(rotation='vertical', fontsize=7)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.2)
    plt.ylabel('AUPRC Score')
    plt.xlabel('Cell lines')
    plot_title = 'AUPRC ' + title_suffix
    plt.title(plot_title, loc='center', wrap=True)

    plot_filename_png = plot_dir + plot_title + '.png'
    plot_filename_pdf = plot_dir + plot_title + '.pdf'

    os.makedirs(os.path.dirname(plot_filename_pdf),exist_ok=True)
    plt.savefig(plot_filename_png)
    plt.savefig(plot_filename_pdf)

    plt.show()
    plt.clf()

    #plot auroc
    plt.scatter(AUROC_sorted_dict.keys(), AUROC_sorted_dict.values(),)
    plt.xticks(rotation='vertical', fontsize=7)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.2)
    plt.ylabel('AUROC Score')
    plt.xlabel('Cell lines')
    plot_title = 'AUROC ' + title_suffix
    plt.title(plot_title, loc = 'center', wrap=True)

    plot_filename_png = plot_dir + plot_title + '.png'
    plot_filename_pdf = plot_dir + plot_title + '.pdf'
    plt.savefig(plot_filename_png)
    plt.savefig(plot_filename_pdf)
    plt.show()
    plt.clf()



def plot_roc_prc(precision, recall, FPR, TPR, alg, cell_line, iter, out_dir):
    ## Make PR curves

    out_dir = out_dir + 'iter' + str(iter) + '/'
    # to_make cell line name compatible for file name replace \/ in the cell line name
    cell_line_file_compat_name = cell_line.replace('/', '_')
    cell_line_file_compat_name = cell_line_file_compat_name.replace('\\', '_')

    os.makedirs(out_dir, exist_ok=True)

    sns.lineplot(recall, precision, ci=None)
    # legendList.append(key + ' (AUPRC = ' + str("%.2f" % (AUPRC)) + ')')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(cell_line + '_iter_' + str(iter))
    try:
        plt.savefig(out_dir + cell_line_file_compat_name + '_PRCurve.pdf')
        plt.savefig(out_dir + cell_line_file_compat_name + '_PRCurve.png')
    except:
        print('problem in auroc/auprc saving with: ', cell_line_file_compat_name)

    if cell_line == 'combo':
        plt.show()
    plt.clf()

    ## Make ROC curves

    sns.lineplot(FPR, TPR, ci=None)
    plt.plot([0, 1], [0, 1], linewidth=1.5, color='k', linestyle='--')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(cell_line + '_iter_' + str(iter))
    try:
        plt.savefig(out_dir + cell_line_file_compat_name + '_ROC.pdf')
        plt.savefig(out_dir + cell_line_file_compat_name + '_ROC.png')
    except:
        print('problem in auroc/auprc saving with: ', cell_line_file_compat_name)

    if cell_line == 'combo':
        plt.show()
    plt.clf()




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


    ##plot the points as well in the boxplot
    # xs = []
    # n=0
    # for cell_line in AUPRC_dict:
    #     xs.append(np.random.normal(n + 1, 0.04, len(plot_data[n])))
    #     n+=1
    # # palette = ['r','g','b','y','o']*int((n)/5)
    # for x, val in zip(xs, plot_data):
    #     plt.scatter(x, val, alpha=0.4, color='r')

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






def plot_pairwise_value_difference(positive_df_all_runs):
    # for pos_df in positive_df_all_runs:
    pos_df= positive_df_all_runs[0]
    pairs_1_2 = pos_df[pos_df['drug_1'] > pos_df['drug_2']]
    pairs_2_1 = pos_df[pos_df['drug_1'] < pos_df['drug_2']]
    assert len(pairs_1_2) == len(pairs_2_1), 'problem non-symmetric'

    pairs_1_2 = pairs_1_2.sort_values(['drug_1','drug_2','cell_line'], ascending=(True, True, True))
    pairs_2_1 = pairs_2_1.sort_values(['drug_2','drug_1','cell_line'], ascending=(True, True,True))
    assert len(pairs_1_2) == len(pairs_2_1), 'problem non-symmetric'

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

def scatter_plot_auc_auprc_all_cell_line_all_runs(AUPRC_dict, AUROC_dict, out_dir):

    #this AUROC_dict and AUPRC_dict is expected to have 2 auroc/ auprc per cell_line. these 2 auprc/auroc score can be from
    #using or not using drug-feature.

    cell_lines = AUPRC_dict.keys()
    auprc_array = np.array(list(AUPRC_dict.values()))


    data_0 = auprc_array[:,0]
    # x_pos = np.arange(len(data_0))
    plt.bar(cell_lines, data_0,align='center', color = 'orange' ,alpha = 0.5)

    data_1 = auprc_array[:, 1]
    plt.bar(cell_lines, data_1, align='center', color= 'blue', alpha=0.5)

    plt.xticks(rotation='vertical', fontsize=7)
    plt.xticks()
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.2)
    plt.xlabel('cell lines')
    plt.ylabel('auprc')
    plt.show()
    plt.close()


def performance_metric_evaluation_per_alg(output_df_per_alg_all_runs, alg, config_map):
    out_dir = config_map['project_dir'] + config_map['output_dir'] + alg + '/' + 'plots/'
    cell_lines = output_df_per_alg_all_runs[0]['cell_line'].unique()

    #performance_metric_per_alg will save dict of dict of dicts: run -> metric -> cell_line -> values.
    # for each run, for each metric, for each cell_line(also 'combo') saves the value of the metric
    performance_metric_per_alg = {iter:{} for iter in range(len(output_df_per_alg_all_runs))}

    #AUPRC_dict, AUROC_dict is a dictionary of lists. each key is a cell_line. each list contains auroc(auprc)\
    # value of that cell line across multiple runs
    AUPRC_dict = {cell_line: [] for cell_line in cell_lines}
    AUROC_dict = {cell_line:[] for cell_line in cell_lines}


    ############################ COMPUTE EVALUATION ###############################
    #computate eval metrics


    iter = 0
    for output_df_per_iter in output_df_per_alg_all_runs:
        precision,recall, FPR, TPR, AUPRC,AUROC = {},{},{},{},{},{}
        output_df_unique_pairs = keep_one_from_symmetric_pairs(output_df_per_iter.copy(), aggregate ='max')
        precision['combo'], recall['combo'], FPR['combo'], TPR['combo'], AUPRC['combo'], AUROC['combo'] \
            = compute_roc_auprc(output_df_unique_pairs)

        #separate the cell lines
        for cell_line in cell_lines:
            cell_line_specific_df = output_df_unique_pairs[output_df_unique_pairs['cell_line'] == cell_line]
            precision[cell_line], recall[cell_line], FPR[cell_line], TPR[cell_line], AUPRC[cell_line], AUROC[cell_line]\
                = compute_roc_auprc(cell_line_specific_df)

            AUPRC_dict[cell_line].append(AUPRC[cell_line])
            AUROC_dict[cell_line].append(AUROC[cell_line])
        #input to the main storage for eval metrics

        performance_metric_per_alg[iter]['precision'] = precision
        performance_metric_per_alg[iter]['recall'] = recall
        performance_metric_per_alg[iter]['FPR'] = FPR
        performance_metric_per_alg[iter]['TPR'] = TPR
        performance_metric_per_alg[iter]['AUPRC'] = AUPRC
        performance_metric_per_alg[iter]['AUROC'] = AUROC

        iter+=1



    ################ PLOT #############################
    #plot per run evaluation

    n_drug_feat_options = 1
    if alg=='decagon':
        decagon_settings = config_map['ml_models_settings']['algs']['decagon']
        use_drug_feat_options = decagon_settings['use_drug_feat']
        n_drug_feat_options = len(use_drug_feat_options)

    for iter in performance_metric_per_alg:
        #plot for comnined evaluation
        key = 'combo'
        plot_roc_prc(performance_metric_per_alg[iter]['precision'][key], \
            performance_metric_per_alg[iter]['recall'][key], performance_metric_per_alg[iter]['FPR'][key],\
            performance_metric_per_alg[iter]['TPR'][key], alg, key , iter, out_dir)

        # plot per cell line evaluation
        for cell_line in cell_lines:
            key = cell_line
            plot_roc_prc(performance_metric_per_alg[iter]['precision'][key], \
                         performance_metric_per_alg[iter]['recall'][key], performance_metric_per_alg[iter]['FPR'][key], \
                         performance_metric_per_alg[iter]['TPR'][key], alg, key, iter, out_dir)

    # plot across all cell lines all runs
    plot_auc_auprc_boxplot_all_cell_line_all_runs(AUPRC_dict, AUROC_dict, out_dir)
    # scatter_plot_auc_auprc_all_cell_line_all_runs(AUPRC_dict, AUROC_dict, out_dir)



# df = pd.DataFrame([[1, 2, 10, 1, 0.6], [2, 1, 10,1, 0.8], [4, 6, 1, 0, 0.4]], \
#                   columns = ["drug_1", "drug_2", "cell_line","true","predicted"])
# df = keep_one_from_symmetric_pairs(df, aggregate = 'max')
