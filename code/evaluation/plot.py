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
import seaborn as sns
from evaluation.utils import keep_one_from_symmetric_pairs, sort_max_drug_first
from evaluation.compute import  compute_roc_prc
from evaluation.utils import EvalScore
from textwrap import wrap

def name_for_plot(alg_name):
    if alg_name=='synverse':
        return 'SynVerse'
    elif alg_name=='synverse_nogenex':
        return 'S_nogenex'
    elif alg_name == 'synverse_v3':
        return 'SynVerse_v3'
    elif alg_name == 'synverse_v4':
        return 'SynVerse_v4'
    elif alg_name == 'synverse_tissuenet':
        return 'SynVerse_multippi'
    elif alg_name=='decagon':
        return 'Decagon'
    elif alg_name=='dtf':
        return 'DTF'
    elif alg_name=='deepsynergy':
        return 'DeepSynergy'

# from evaluation.utils import EvalScore, set_title_suffix
################## input: single model instance, plot: overall performance#########################

def plot_predicted_score_distribution(df, title_suffix, plot_dir):
    # input is from a single run for a single algorithm.

    pos_df = df[df['true'] == 1]
    neg_df = df[df['true'] == 0]
    # pos_df_unique_pairs = keep_one_from_symmetric_pairs(pos_df.copy(), aggregate='max')
    # neg_df_unique_pairs = keep_one_from_symmetric_pairs(neg_df.copy(), aggregate='max')
    pos_data = list(pos_df['predicted'])
    neg_data = list(neg_df['predicted'])
    print(len(pos_data), len(neg_data))
    plt.hist(pos_data, weights=np.zeros_like(pos_data) + 1. / len(pos_data),
             alpha=0.5, bins=30, color='blue', range=(0, 1), label='positive')
    plt.hist(neg_data, weights=np.zeros_like(neg_data) + 1. / len(neg_data),
             alpha=0.5, bins=30, color='orange', range=(0, 1), label='negative')

    plt.xlabel('predicted score')
    plt.ylabel('fraction of total (pos or neg) pairs')

    file_title = 'score_distribution_' + title_suffix
    plot_filename_png = plot_dir + file_title + '.png'
    plot_filename_pdf = plot_dir + file_title + '.pdf'

    plot_title = '\n'.join(wrap(file_title, 60))
    plt.title(plot_title, loc = 'left')
    plt.tight_layout(pad=1, w_pad=0.4, h_pad=0.5)

    plt.legend()



    os.makedirs(os.path.dirname(plot_filename_pdf), exist_ok=True)
    # plt.tight_layout()
    plt.tight_layout(pad=1, w_pad=0.4, h_pad=0.5)

    print('file name: ', plot_filename_png )
    plt.savefig(plot_filename_png)
    plt.savefig(plot_filename_pdf)

    # plt.savefig(plot_filename_png, bbox_inches='tight')
    # plt.savefig(plot_filename_pdf,  bbox_inches='tight')

    plt.show()
    # plt.clf()
    plt.close()




################## input: prediction score single model instance, single run
################## plot: cell line wise evaluation score of a model #########################
def scatter_plot_auprc_auroc_single_run(AUPRC, AUROC, title_suffix, plot_dir):

    # df = keep_one_from_symmetric_pairs(df.copy(), aggregate='max')

    # _, _, _, _, auprc, auc = compute_roc_prc(df)
    # print('all cell lines: AUPRC, AUROC', auprc, auc)

    AUPRC_sorted_dict = dict(sorted(AUPRC.items(), key=lambda item: item[1], reverse=True))
    AUROC_sorted_dict = dict(sorted(AUROC.items(), key=lambda item: item[1], reverse=True))

    #plot auprc
    plt.scatter(AUPRC_sorted_dict.keys(), AUPRC_sorted_dict.values())
    plt.xticks(rotation='vertical', fontsize=5)
    # plt.margins(0.2)
    # plt.subplots_adjust(bottom=0.2)
    plt.ylabel('AUPRC Score')
    plt.xlabel('Cell lines')
    # plot_title = 'AUPRC_' + title_suffix
    # plt.title(plot_title, loc='center', wrap=True)

    file_title = 'AUPRC_' + title_suffix
    plot_filename_png = plot_dir + file_title + '.png'
    plot_filename_pdf = plot_dir + file_title + '.pdf'

    plot_title = '\n'.join(wrap(file_title, 50))
    plt.title(plot_title, loc='left')
    plt.tight_layout(pad=1, w_pad=0.4, h_pad=0.5)



    os.makedirs(os.path.dirname(plot_filename_pdf),exist_ok=True)
    plt.savefig(plot_filename_png,  bbox_inches='tight')
    plt.savefig(plot_filename_pdf,  bbox_inches='tight')

    plt.show()
    # plt.clf()
    plt.close()

    #plot auroc
    plt.scatter(AUROC_sorted_dict.keys(), AUROC_sorted_dict.values(),)
    plt.xticks(rotation='vertical', fontsize=5)
    # plt.margins(0.2)
    # plt.subplots_adjust(bottom=0.2)
    plt.ylabel('AUROC Score')
    plt.xlabel('Cell lines')
    # plot_title = 'AUROC_' + title_suffix
    # plt.title(plot_title, loc = 'center', wrap=True)

    file_title = 'AUROC_' + title_suffix
    plot_filename_png = plot_dir + file_title + '.png'
    plot_filename_pdf = plot_dir + file_title + '.pdf'

    plot_title = '\n'.join(wrap(file_title, 50))
    plt.title(plot_title, loc='left')
    plt.tight_layout(pad=1, w_pad=0.4, h_pad=0.5)

    plt.savefig(plot_filename_png,  bbox_inches='tight')
    plt.savefig(plot_filename_pdf,  bbox_inches='tight')
    plt.show()
    # plt.clf()
    plt.close()

    return AUPRC, AUROC


def box_plot_auprc_auroc_all_runs(auprc_all_runs, auroc_all_runs, title_suffix, \
                                  cell_line_condition, plot_dir):
    '''
    This function will plot auprc and auroc score for each cell line separately
    across all the runs for a certain algorithm with a certain param setting.
    input: auprc_all_runs: dict of dict. The keys for outer dict are the run numbers.
    the keys for inner dictionary are the cell_line names. So, this contains auprc scores for each
    cell line across all runs for a ceratin algo with certain param setting.
    same goes for auroc_all_runs.
    '''

    #plot AUPRC
    if cell_line_condition=='sep':
        auprc_all_runs_df = pd.DataFrame(auprc_all_runs)
    elif cell_line_condition=='combo':
        auprc_all_runs_df = pd.DataFrame(auprc_all_runs, index = [0])
    auprc_all_runs_df = auprc_all_runs_df.T

    sns.boxplot(data=auprc_all_runs_df, whis=[0,1])


    title_suffix = title_suffix.replace('use_','')
    title_suffix = title_suffix.replace('sampling','')

    file_title = 'AUPRC_all_runs_' + cell_line_condition + '_'+title_suffix
    plt.xticks(rotation='vertical', fontsize=5)
    plot_title = '\n'.join(wrap(file_title, 50))
    plt.title(plot_title, loc='left')
    plt.tight_layout(pad=1, w_pad=0.4, h_pad=0.5)

    plot_filename = plot_dir + file_title + '.pdf'
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()
    # plt.clf()
    plt.close()

    # plot AUROC
    if cell_line_condition=='sep':
        auroc_per_cell_line_all_runs_df = pd.DataFrame(auroc_all_runs)
    elif cell_line_condition =='combo':
        auroc_per_cell_line_all_runs_df = pd.DataFrame(auroc_all_runs, index=[0])
    auroc_per_cell_line_all_runs_df = auroc_per_cell_line_all_runs_df.T

    sns.boxplot(data=auroc_per_cell_line_all_runs_df, whis=[0,1])

    file_title = 'AUROC_all_runs_' + cell_line_condition + '_'+title_suffix
    plt.xticks(rotation='vertical', fontsize=5)
    plot_title = '\n'.join(wrap(file_title, 50))
    plt.title(plot_title, loc='left')
    plt.tight_layout(pad=1, w_pad=0.4, h_pad=0.5)

    plot_filename = plot_dir + file_title + '.pdf'

    plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()
    # plt.clf()
    plt.close()




def plot_roc_prc(precision, recall, FPR, TPR, title_suffix, plot_dir):
    ## Make PR curves

    sns.lineplot(recall, precision, ci=None)
    # legendList.append(key + ' (AUPRC = ' + str("%.2f" % (AUPRC)) + ')')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plot_title = 'PRcurve_' + title_suffix
    # plt.title(plot_title, loc='center', wrap=True)

    file_title = 'PRcurve_' + title_suffix
    plot_filename_png = plot_dir + file_title + '.png'
    plot_filename_pdf = plot_dir + file_title + '.pdf'

    plot_title = '\n'.join(wrap(file_title, 50))
    plt.title(plot_title, loc='left')
    plt.tight_layout(pad=1, w_pad=0.4, h_pad=0.5)


    plt.savefig(plot_filename_png,  bbox_inches='tight')
    plt.savefig(plot_filename_pdf,  bbox_inches='tight')

    plt.show()
    plt.close()

    ## Make ROC curves

    sns.lineplot(FPR, TPR, ci=None)
    # legendList.append(key + ' (AUPRC = ' + str("%.2f" % (AUPRC)) + ')')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # plot_title = 'ROCcurve_' + title_suffix
    # plt.title(plot_title, loc='center', wrap=True)
    file_title = 'ROCcurve_' + title_suffix
    plot_filename_png = plot_dir + file_title + '.png'
    plot_filename_pdf = plot_dir + file_title + '.pdf'

    plot_title = '\n'.join(wrap(file_title, 50))
    plt.title(plot_title, loc='left')
    plt.tight_layout(pad=1, w_pad=0.4, h_pad=0.5)


    plt.savefig(plot_filename_png, bbox_inches='tight')
    plt.savefig(plot_filename_pdf, bbox_inches='tight')

    plt.show()
    plt.close()

def plot_best_models_auprc_auroc_e_prec(early_prec_k,neg_fact, eval_score_dict, plot_dir):
    #eval_score_dict[alg] => is a list containing the performance of best-param-setting-model of 'alg'. This is a
    # list because it contains score from multiple runs
    auprc_data_all_alg = []
    auroc_data_all_alg = []
    e_prec_data_all_alg = []
    plot_label = []
    for alg in eval_score_dict:
        auprc_data = []
        auroc_data = []
        e_prec_data = []
        for run_ in range(len(eval_score_dict[alg])):
            auprc_data.append(eval_score_dict[alg][run_].auprc)
            auroc_data.append(eval_score_dict[alg][run_].auroc)
            e_prec_data.append(eval_score_dict[alg][run_].early_prec)
        auprc_data_all_alg.append(auprc_data)
        auroc_data_all_alg.append(auroc_data)
        e_prec_data_all_alg.append(e_prec_data)

        alg_name_for_plot = name_for_plot(alg)
        plot_label.append(alg_name_for_plot)

        print('Parameter for best model: ' + alg)
    fig, ax = plt.subplots()
    ax.margins(x=0)

    plt.boxplot(auprc_data_all_alg, labels=plot_label)
    print('AUPRC: ', auprc_data_all_alg)
    # bplot = plt.boxplot(auprc_data_all_alg, labels=plot_label,
    #                     patch_artist=True, widths=0.7, positions = [0,1,2])
    #
    # colors = ['pink', 'lightblue', 'lightgreen']
    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)

    y_val = 1 / float(1 + neg_fact)
    plt.axhline(y=y_val, color="gray", linestyle="--")
    ax.set_ylim([0.5, 0.85])

    plt.xlabel('Algorithms', fontsize=11)
    plt.xticks(fontsize=8.5)
    plt.ylabel('AUPRC', fontsize=11)
    plt.title('Comparing AUPRC', fontsize=12)
    # plt.tight_layout()
    plt.tight_layout()

    plt.savefig(plot_dir + 'compare_AUPRC.png', bbox_inches='tight')
    plt.savefig(plot_dir + 'compare_AUPRC.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    print(plot_dir + 'compare_AUPRC.png')

    #AUROC
    fig, ax = plt.subplots()
    plt.boxplot(auroc_data_all_alg, labels=plot_label)
    # bplot = plt.boxplot(auroc_data_all_alg, labels=plot_label, patch_artist=True)
    # colors = ['pink', 'lightblue', 'lightgreen']
    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)

    y_val = 0.5
    plt.axhline(y=y_val, color="gray", linestyle="--")
    ax.set_ylim([0.5, 0.85])
    plt.xlabel('Algorithms')
    plt.ylabel('AUROC')
    plt.title('Comparing AUROC of different algorithms')

    plt.tight_layout()
    plt.savefig(plot_dir+'compare_AUROC.png', bbox_inches='tight')
    plt.savefig(plot_dir + 'compare_AUROC.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    #Early Prec
    fig, ax = plt.subplots()
    plt.boxplot(e_prec_data_all_alg, labels=plot_label)
    # bplot = plt.boxplot(e_prec_data_all_alg, labels=plot_label, patch_artist=True)
    # colors = ['pink', 'lightblue', 'lightgreen']
    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)
    #plot baseline
    y_val = 1/float(1+neg_fact)
    plt.axhline(y=y_val, color="gray", linestyle="--")
    ax.set_ylim([0.5,0.85])
    plt.xlabel('Algorithms')
    plt.ylabel('Early_precision_at '+ str(early_prec_k))
    plt.title('comparing early precision of different algorithms')

    plt.tight_layout()
    plt.savefig(plot_dir + 'compare_early_prec_'+str(early_prec_k).replace('.','_') +'.png', bbox_inches='tight')
    plt.savefig(plot_dir + 'compare_early_prec_'+str(early_prec_k).replace('.','_') +'.pdf', bbox_inches='tight')
    plt.show()
    plt.close()





########### input: performance score from multiple instances/run/ of one algorithm. in the input dictionary the auprc/auroc
# score are both for a prediction on a   cell line  (under the key=cell line name) and
# for over all cell lines prediction(i.e. across all cell lines under 'combo' key)

# def plot_auc_auprc_boxplot_all_cell_line_all_runs(AUPRC_dict, AUROC_dict,out_dir ):
#
#     #plot AUPRC boxplot
#     plot_data=[]
#     plot_label=[]
#
#     AUPRC_med_dict = {cell_line: statistics.median(AUPRC_dict[cell_line]) for cell_line in AUPRC_dict}
#
#     AUPRC_med_sorted_dict = dict(sorted(AUPRC_med_dict.items(), key=lambda item: item[1], reverse=True))
#
#     for cell_line in AUPRC_med_sorted_dict:
#         plot_data.append(AUPRC_dict[cell_line])
#         plot_label.append(cell_line)
#     plt.boxplot(plot_data,labels=plot_label)
#
#
#     ##plot the points as well in the boxplot
#     # xs = []
#     # n=0
#     # for cell_line in AUPRC_dict:
#     #     xs.append(np.random.normal(n + 1, 0.04, len(plot_data[n])))
#     #     n+=1
#     # # palette = ['r','g','b','y','o']*int((n)/5)
#     # for x, val in zip(xs, plot_data):
#     #     plt.scatter(x, val, alpha=0.4, color='r')
#
#     plt.xticks(rotation='vertical', fontsize=7)
#     plt.margins(0.2)
#     plt.subplots_adjust(bottom=0.2)
#     plt.ylabel('AUPRC Score')
#     plt.xlabel('Cell lines')
#
#     plt.savefig(out_dir+'auprc_across_all_runs.png')
#     plt.savefig(out_dir + 'auprc_across_all_runs.pdf')
#     plt.show()
#     plt.clf()
#
#     # plot AUROC boxplot
#     plot_data = []
#     plot_label = []
#     AUROC_med_dict = {cell_line: statistics.median(AUROC_dict[cell_line]) for cell_line in AUROC_dict}
#     AUROC_med_sorted_dict = dict(sorted(AUROC_med_dict.items(), key=lambda item: item[1], reverse=True))
#     for cell_line in AUROC_med_sorted_dict:
#         plot_data.append(AUROC_dict[cell_line])
#         plot_label.append(cell_line)
#     plt.boxplot(plot_data, labels=plot_label)
#
#     plt.xticks(rotation='vertical', fontsize=7)
#     plt.margins(0.2)
#     plt.subplots_adjust(bottom=0.2)
#     plt.ylabel('AUROC Score')
#     plt.xlabel('Cell lines')
#
#     plt.savefig(out_dir + 'auroc_across_all_runs.png')
#     plt.savefig(out_dir + 'auroc_across_all_runs.pdf')
#     plt.show()
#     plt.clf()
#
#




# def plot_pairwise_value_difference(positive_df_all_runs):
#     # for pos_df in positive_df_all_runs:
#     pos_df= positive_df_all_runs[0]
#     pairs_1_2 = pos_df[pos_df['drug_1'] > pos_df['drug_2']]
#     pairs_2_1 = pos_df[pos_df['drug_1'] < pos_df['drug_2']]
#     assert len(pairs_1_2) == len(pairs_2_1), 'problem non-symmetric'
#
#     pairs_1_2 = pairs_1_2.sort_values(['drug_1','drug_2','cell_line'], ascending=(True, True, True))
#     pairs_2_1 = pairs_2_1.sort_values(['drug_2','drug_1','cell_line'], ascending=(True, True,True))
#     assert len(pairs_1_2) == len(pairs_2_1), 'problem non-symmetric'
#
#     diff_data = np.array(pairs_1_2['predicted'].values)-np.array(pairs_2_1['predicted'].values)
#     diff_data = np.absolute(diff_data)
#
#     plt.boxplot(diff_data, labels=['diff'], autorange=True)
#     plt.title('difference between scores of (x,y) and (y,x) pairs')
#     plt.show()
#     plt.clf()
#
#     bin_seq = np.linspace(0,0.05,20)
#     # plt.hist(diff_data)
#     plt.hist(diff_data,\
#              weights=np.zeros_like(diff_data) + 1. / diff_data.size,bins=bin_seq, alpha=0.7)
#     plt.xlabel('difference between scores of (x,y) and (y,x)')
#     plt.ylabel('fraction of total pairs')
#     plt.show()
#     plt.clf()
#     data_stat = [i for i in diff_data if abs(i) < 0.01]
#     print(len(data_stat), len(diff_data))

# def scatter_plot_auc_auprc_all_cell_line_all_runs(AUPRC_dict, AUROC_dict, out_dir):
#
#     #this AUROC_dict and AUPRC_dict is expected to have 2 auroc/ auprc per cell_line. these 2 auprc/auroc score can be from
#     #using or not using drug-feature.
#
#     cell_lines = AUPRC_dict.keys()
#     auprc_array = np.array(list(AUPRC_dict.values()))
#
#
#     data_0 = auprc_array[:,0]
#     # x_pos = np.arange(len(data_0))
#     plt.bar(cell_lines, data_0,align='center', color = 'orange' ,alpha = 0.5)
#
#     data_1 = auprc_array[:, 1]
#     plt.bar(cell_lines, data_1, align='center', color= 'blue', alpha=0.5)
#
#     plt.xticks(rotation='vertical', fontsize=7)
#     plt.xticks()
#     plt.margins(0.2)
#     plt.subplots_adjust(bottom=0.2)
#     plt.xlabel('cell lines')
#     plt.ylabel('auprc')
#     plt.show()
#     plt.close()

####################### input:evaluation score for a cell line or evaluation score  over all cell lines from a single model instance,
##################### plot: according to input (for a cell line or over all cell lines)
# def plot_roc_prc(precision, recall, FPR, TPR, alg, cell_line, iter, out_dir):
#     ## Make PR curves
#
#     out_dir = out_dir + 'iter' + str(iter) + '/'
#     # to_make cell line name compatible for file name replace \/ in the cell line name
#     cell_line_file_compat_name = cell_line.replace('/', '_')
#     cell_line_file_compat_name = cell_line_file_compat_name.replace('\\', '_')
#
#     os.makedirs(out_dir, exist_ok=True)
#
#     sns.lineplot(recall, precision, ci=None)
#     # legendList.append(key + ' (AUPRC = ' + str("%.2f" % (AUPRC)) + ')')
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title(cell_line + '_iter_' + str(iter))
#     try:
#         plt.savefig(out_dir + cell_line_file_compat_name + '_PRCurve.pdf')
#         plt.savefig(out_dir + cell_line_file_compat_name + '_PRCurve.png')
#     except:
#         print('problem in auroc/auprc saving with: ', cell_line_file_compat_name)
#
#     if cell_line == 'combo':
#         plt.show()
#     plt.clf()
#
#     ## Make ROC curves
#
#     sns.lineplot(FPR, TPR, ci=None)
#     plt.plot([0, 1], [0, 1], linewidth=1.5, color='k', linestyle='--')
#
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.xlabel('FPR')
#     plt.ylabel('TPR')
#     plt.title(cell_line + '_iter_' + str(iter))
#     try:
#         plt.savefig(out_dir + cell_line_file_compat_name + '_ROC.pdf')
#         plt.savefig(out_dir + cell_line_file_compat_name + '_ROC.png')
#     except:
#         print('problem in auroc/auprc saving with: ', cell_line_file_compat_name)
#
#     if cell_line == 'combo':
#         plt.show()
#     plt.clf()

# def performance_metric_evaluation_per_alg(output_df_per_alg_all_runs, alg, config_map):
#     out_dir = config_map['project_dir'] + config_map['output_dir'] + alg + '/' + 'plots/'
#     cell_lines = output_df_per_alg_all_runs[0]['cell_line'].unique()
#
#     #performance_metric_per_alg will save dict of dict of dicts: run -> metric -> cell_line -> values.
#     # for each run, for each metric, for each cell_line(also 'combo') saves the value of the metric
#     performance_metric_per_alg = {iter:{} for iter in range(len(output_df_per_alg_all_runs))}
#
#     #AUPRC_dict, AUROC_dict is a dictionary of lists. each key is a cell_line. each list contains auroc(auprc)\
#     # value of that cell line across multiple runs
#     AUPRC_dict = {cell_line: [] for cell_line in cell_lines}
#     AUROC_dict = {cell_line:[] for cell_line in cell_lines}
#
#
#     ############################ COMPUTE EVALUATION ###############################
#     #computate eval metrics
#
#
#     iter = 0
#     for output_df_per_iter in output_df_per_alg_all_runs:
#         precision,recall, FPR, TPR, AUPRC,AUROC = {},{},{},{},{},{}
#         output_df_unique_pairs = keep_one_from_symmetric_pairs(output_df_per_iter.copy(), aggregate ='max')
#         precision['combo'], recall['combo'], FPR['combo'], TPR['combo'], AUPRC['combo'], AUROC['combo'] \
#             = compute_roc_prc(output_df_unique_pairs)
#
#         #separate the cell lines
#         # for cell_line in cell_lines:
#         #     cell_line_specific_df = output_df_unique_pairs[output_df_unique_pairs['cell_line'] == cell_line]
#         #     precision[cell_line], recall[cell_line], FPR[cell_line], TPR[cell_line], AUPRC[cell_line], AUROC[cell_line]\
#         #         = compute_roc_prc(cell_line_specific_df)
#         #
#         #     AUPRC_dict[cell_line].append(AUPRC[cell_line])
#         #     AUROC_dict[cell_line].append(AUROC[cell_line])
#         #input to the main storage for eval metrics
#
#         performance_metric_per_alg[iter]['precision'] = precision
#         performance_metric_per_alg[iter]['recall'] = recall
#         performance_metric_per_alg[iter]['FPR'] = FPR
#         performance_metric_per_alg[iter]['TPR'] = TPR
#         performance_metric_per_alg[iter]['AUPRC'] = AUPRC
#         performance_metric_per_alg[iter]['AUROC'] = AUROC
#
#         iter+=1
#
#
#
#     ################ PLOT #############################
#     #plot per run evaluation
#
#     n_drug_feat_options = 1
#     if alg=='decagon':
#         decagon_settings = config_map['ml_models_settings']['algs']['decagon']
#         use_drug_feat_options = decagon_settings['use_drug_feat']
#         n_drug_feat_options = len(use_drug_feat_options)
#
#     for iter in performance_metric_per_alg:
#         #plot for comnined evaluation
#         key = 'combo'
#         plot_roc_prc(performance_metric_per_alg[iter]['precision'][key], \
#             performance_metric_per_alg[iter]['recall'][key], performance_metric_per_alg[iter]['FPR'][key],\
#             performance_metric_per_alg[iter]['TPR'][key], alg, key , iter, out_dir)
#
#         # plot per cell line evaluation
#         for cell_line in cell_lines:
#             key = cell_line
#             plot_roc_prc(performance_metric_per_alg[iter]['precision'][key], \
#                          performance_metric_per_alg[iter]['recall'][key], performance_metric_per_alg[iter]['FPR'][key], \
#                          performance_metric_per_alg[iter]['TPR'][key], alg, key, iter, out_dir)
#
#     # plot across all cell lines all runs
#     # plot_auc_auprc_boxplot_all_cell_line_all_runs(AUPRC_dict, AUROC_dict, out_dir)
#     # scatter_plot_auc_auprc_all_cell_line_all_runs(AUPRC_dict, AUROC_dict, out_dir)



# df = pd.DataFrame([[1, 2, 10, 1, 0.6], [2, 1, 10,1, 0.8], [4, 6, 1, 0, 0.4]], \
#                   columns = ["drug_1", "drug_2", "cell_line","true","predicted"])
# df = keep_one_from_symmetric_pairs(df, aggregate = 'max')
