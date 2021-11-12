import os

import evaluation.compute as compute
import pandas as pd
import argparse
import yaml
import sys
import numpy as np
from evaluation.utils import EvalScore, set_title_suffix, keep_one_from_symmetric_pairs
import evaluation.plot as eval_plot
sys.path.insert(0, '/home/tasnina/Projects/Synverse/')
from utils import *
import re
from evaluation.compute import  compute_roc_prc
from evaluation.compute import  *

def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        # config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.load(conf)
    # TODO check to make sure the inputs are correct in config_map
    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="""Script to download and parse input files,
                                     and (TODO) run the  pipeline using them.""")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str,
                       default="/home/tasnina/Projects/SynVerse/code/config-files/master_config_csbgpu3.yaml",
                       help="Configuration file for this script.")

    group.add_argument('--recall', type=float,
                       default=0.3,
                       help="recall value for early precision")

    return parser




# def get_performance_scores(pos_neg_df, early_prec_recall_val):
#     prec, recall, fpr, tpr, auprc, auroc = compute.compute_roc_prc(pos_neg_df)
#     early_prec = compute.compute_early_prec(pos_neg_df, early_prec_recall_val)
#     return auprc, auroc, early_prec


def create_model_info_string(alg_name, params):
    if (alg_name=='synverse')| (alg_name=='synverse_v2')|(alg_name=='synverse_v3')|(alg_name=='synverse_v4')|\
            (alg_name=='synverse_nogenex'):
        model_param = {
            'encoder': params['encoder'], 'decoder': params['decoder'],
            'hsize': params['hsize'], 'nn_hsize': params['nn_hsize'],
            'drugfeat': params['drugfeat'],
            'e': params['e'], 'lr': params['lr'], 'batch': params['batch'],
            'dr': params['dr']
        }


        model_info_str = ''
        for key in model_param:
            model_info_str = model_info_str + '_' + key + '_' + str(model_param[key])

        # delete the phrase 'nn_hsize' as I did not use it while saving predction in synverse run
        model_info_str = model_info_str.replace('_nn_hsize', '')
    if (alg_name=='synverse_tissuenet'):
        model_param = {
            'encoder': params['encoder'], 'decoder': params['decoder'], 'gg_decoder':params['gg_decoder'],
            'hsize': params['hsize'], 'nn_hsize': params['nn_hsize'],
            'drugfeat': params['drugfeat'],
            'e': params['e'], 'lr': params['lr'], 'batch': params['batch'],
            'dr': params['dr']
        }


        model_info_str = ''
        for key in model_param:
            model_info_str = model_info_str + '_' + key + '_' + str(model_param[key])

        # delete the phrase 'nn_hsize' as I did not use it while saving predction in synverse run
        model_info_str = model_info_str.replace('_nn_hsize', '')
        model_info_str = model_info_str.replace('_gg_decoder', '')


    elif alg_name=='decagon':
        model_param = {
            'encoder': params['encoder'], 'decoder': params['decoder'],
            'hsize': params['hsize'],
            'drugfeat': params['drugfeat'],
            'e': params['e'], 'lr': params['lr'], 'batch': params['batch'],
            'dr': params['dr']
        }

        model_info_str = ''
        for key in model_param:
            model_info_str = model_info_str + '_' + key + '_' + str(model_param[key])
    elif alg_name=='deepsynergy':
        input_dropout = params['dr'][0]
        dropout = params['dr'][1]

        model_param = {'layers': params['layers'], 'e': params['e'], 'lr': params['lr'],
                       'batch': params['batch'], 'dr': input_dropout, 'dr_hid': dropout, 'norm': params['norm'],
                       'act': params['act'], 'use_genex': params['use_genex'], 'use_target': params['use_target'],
                       'reduce_dim': params['reduce_dim']}

        model_info_str = ''
        for key in model_param:
            model_info_str = model_info_str + '_' + key + '_' + str(model_param[key])
        model_info_str = model_info_str.replace('_dr_hid', '')
    elif alg_name=='dtf':
        dr_str = '_'.join(str(x) for x in params['dr'])

        model_param = {'layers': params['nn_struct'],
                       'lr': params['learning_rate'],
                       'dr': dr_str,
                       'norm': params['norm']}

        model_info_str = ''
        for key in model_param.keys():
            model_info_str = model_info_str + '_' + key + '_' + str(model_param[key])


    return model_param, model_info_str

def get_pos_neg_predictions_files(should_run_algs,  param_settings_dict):
    model_params = {alg: [] for alg in
                    should_run_algs}  # dictionary of  list of dictionaries. settings for each alg will be kept in the innermost dictionary
    pos_neg_file_names = {alg: [] for alg in
                          should_run_algs}  # dictionary of list of tuples. each tupe contain two strings\
    # for positive and negative file names
    for alg in should_run_algs:
        if (alg == 'synverse')|(alg=='synverse_v2')|(alg=='synverse_v3')|(alg=='synverse_v4')|\
                (alg == 'synverse_nogenex')| (alg == 'synverse_tissuenet'):
            synverse_param_settings_list = param_settings_dict[alg]  # all considered values for each param is here
            for params in synverse_param_settings_list:
                # model parameter and model output files
                model_param, model_info_str =  create_model_info_string(alg, params)
                model_params[alg].append(model_param)
                pos_out_file = '/pos_val_scores' + model_info_str + '.tsv'
                neg_out_file = '/neg_val_scores' + model_info_str + '.tsv'
                pos_neg_file_names[alg].append((pos_out_file, neg_out_file))

        if alg == 'decagon':
            decagon_param_settings_list = param_settings_dict[alg]  # all considered values for each param is here
            for params in decagon_param_settings_list:
                # model parameter and model output files
                model_param, model_info_str = create_model_info_string(alg, params)
                model_params[alg].append(model_param)
                pos_out_file = '/pos_val_scores' + model_info_str + '.tsv'
                neg_out_file = '/neg_val_scores' + model_info_str + '.tsv'
                pos_neg_file_names[alg].append((pos_out_file, neg_out_file))

        elif alg == 'deepsynergy':
            ds_param_settings_list = param_settings_dict[alg]
            for params in ds_param_settings_list:
                model_param, model_info_str = create_model_info_string(alg, params)
                model_params[alg].append(model_param)
                pos_out_file = '/pos_val_scores' + model_info_str + '.tsv'
                neg_out_file = '/neg_val_scores' + model_info_str + '.tsv'
                pos_neg_file_names[alg].append((pos_out_file, neg_out_file))

        elif alg == 'dtf':
            dtf_param_settings_list = param_settings_dict[alg]

            for params in dtf_param_settings_list:
                model_param, model_info_str = create_model_info_string(alg, params)
                model_params[alg].append(model_param)
                pos_out_file = '/pos_scores' + model_info_str + '.tsv'
                neg_out_file = '/neg_scores' + model_info_str + '.tsv'

                # model parameter and model output files
                pos_neg_file_names[alg].append((pos_out_file, neg_out_file))
    return pos_neg_file_names, model_params
def get_data_preprocess_params(cross_val_type, config_map, kwargs):
    neg_sampling_type = kwargs.get('sampling')

    neg_fact = config_map['ml_models_settings']['cross_val']['neg_fact']
    val_frac = config_map['ml_models_settings']['cross_val']['val_frac']


    min_pairs_per_cell_line = config_map['synergy_data_settings']['min_pairs']
    max_pairs_per_cell_line = config_map['synergy_data_settings']['max_pairs']
    threshold = config_map['synergy_data_settings']['threshold']['val']
    number_of_top_cell_lines = config_map['synergy_data_settings']['number_of_top_cell_lines']
    number_of_test_cell_lines = config_map['synergy_data_settings']['number_of_test_cell_lines']
    top_k_percent = config_map['synergy_data_settings']['top_k_percent_pairs']


    # this is used in plot file names and title
    dataset_params = {'min_pairs': min_pairs_per_cell_line,
                      'max_pairs': max_pairs_per_cell_line,
                      'th': threshold,
                      'cell_lines': number_of_top_cell_lines, 'percent': top_k_percent,
                      'neg': neg_fact, 'val_frac': val_frac, 'sampling': neg_sampling_type}

    if cross_val_type!= 'test_rare_cell_lines':
        preprocess_params_str = cross_val_type + '/' + 'pairs_' + str(min_pairs_per_cell_line) + '_' + \
                                str(max_pairs_per_cell_line) + '_th_' + str(threshold) + '_cell_lines_' + str(
            number_of_top_cell_lines) + \
                                '_percent_' + str(top_k_percent) + '_' + 'neg_' + str(neg_fact) + '_neg_sampling_' + \
                                neg_sampling_type + '_val_frac_' + str(val_frac) + '_' + kwargs.get('cvdir') + '/'
    else:
        preprocess_params_str = cross_val_type + '/' + 'pairs_' + str(min_pairs_per_cell_line) + '_' + \
                                str(max_pairs_per_cell_line) + '_th_' + str(threshold) + '_cell_lines_' + str(
            number_of_top_cell_lines) +'_'+str(number_of_test_cell_lines) +\
                                '_percent_' + str(top_k_percent) + '_' + 'neg_' + str(neg_fact) + '_neg_sampling_' + \
                                neg_sampling_type + '_val_frac_' + str(val_frac) + '_' + kwargs.get('cvdir') + '/'

    return dataset_params, preprocess_params_str

def evaluate(should_run_algs,param_settings_dict, cross_val_type, kwargs, config_map):
    # get settings from config map
    dataset_params, preprocess_params_str = get_data_preprocess_params(cross_val_type, config_map, kwargs)
    number_of_runs = config_map['ml_models_settings']['runs']
    early_prec_k = kwargs.get('recall')
    neg_fact = config_map['ml_models_settings']['cross_val']['neg_fact']


    # preds_from_best_models = {alg: pd.DataFrame() for alg in should_run_algs}
    best_avg_auprc = {alg: 0 for alg in should_run_algs}
    best_avg_auroc = {alg: 0 for alg in should_run_algs}


    #get prediction file names for each model and settings
    pos_neg_file_names, model_params = get_pos_neg_predictions_files(should_run_algs, param_settings_dict)

    #plot precition of each model (i.e. for each parameter setup at each run.)
    #also find one best model for each alg
    best_models_eval_score_dict = {alg: [] for alg in should_run_algs}
    best_model_param_dict = {alg: {} for alg in should_run_algs}


    for alg in should_run_algs:
        i = 0
        for pos_out_file, neg_out_file in pos_neg_file_names[alg]:
            sum_auprc = 0
            eval_scores = []
            # sum_auroc = 0

            auprc_per_cell_line_all_runs = {}
            auroc_per_cell_line_all_runs={}

            auprc_all_runs = {}
            auroc_all_runs={}
            for run_ in range(number_of_runs):

                out_dir = config_map['project_dir'] + config_map['output_dir']
                alg_spec_out_dir = alg + '/' + preprocess_params_str
                alg_spec_out_dir_per_run = alg_spec_out_dir+ '/'+'run_' + str(run_) + '/'
                alg_spec_out_path_per_run = out_dir + alg_spec_out_dir_per_run

                print('pos file name: ', alg_spec_out_dir_per_run + pos_out_file )

                if alg!='dtf':
                    pos_df = pd.read_csv(alg_spec_out_path_per_run + pos_out_file, sep='\t')
                    neg_df = pd.read_csv(alg_spec_out_path_per_run + neg_out_file, sep='\t')
                else:
                    #handling the inconsistency generated in dtf pipeline while saving predictions
                    pos_df = pd.read_csv(alg_spec_out_path_per_run + pos_out_file)
                    neg_df = pd.read_csv(alg_spec_out_path_per_run + neg_out_file)
                    pos_df['predicted'] = pos_df['predicted'].apply(lambda x: re.split(r'[\[\]]', x)[1]).astype(np.float32)
                    neg_df['predicted'] = neg_df['predicted'].apply(lambda x: re.split(r'[\[\]]', x)[1]).astype(np.float32)

                pos_neg_df = pd.concat([pos_df, neg_df], axis=0)\
                    [['drug_1', 'drug_2', 'cell_line', 'predicted', 'true']]
                pos_neg_df = keep_one_from_symmetric_pairs(pos_neg_df, aggregate='max')


                prec, recall, fpr, tpr, auprc, auroc = compute.compute_roc_prc(pos_neg_df)
                auprc_all_runs[run_], auroc_all_runs[run_] = auprc, auroc

                e_prec_saving_file = alg_spec_out_path_per_run + '.'.join(pos_out_file.split('.')[0:-1]). \
                    replace('pos_val_scores', 'early_prec') + '.tsv'
                e_prec_saving_file = e_prec_saving_file.replace('pos_scores','early_prec')
                early_prec = compute.compute_early_prec(pos_neg_df, early_prec_k, e_prec_saving_file, True)

                eval_scores.append(EvalScore(auprc, auroc, early_prec))
                sum_auprc += auprc

                plot_dir = out_dir + 'Viz/'+ alg_spec_out_dir_per_run
                plot_dir = plot_dir

                os.makedirs(plot_dir, exist_ok=True)

                title_suffix = alg+'_run_' + str(run_) + set_title_suffix(dataset_params, model_params[alg][i])
                #plot distribution of predicted values for positive and negative drug-drug pairs
                # eval_plot.plot_predicted_score_distribution(pos_neg_df, title_suffix, plot_dir)


                auprc_per_cell_line_all_runs[run_], auroc_per_cell_line_all_runs[run_] = \
                                                                            compute_auprc_auroc_per_cell_line(pos_neg_df)

                # plot auprc and auroc per cell line per model per parameter settings
                # eval_plot.scatter_plot_auprc_auroc_single_run(auprc_per_cell_line_all_runs[run_],
                #                                               auroc_per_cell_line_all_runs[run_], title_suffix, plot_dir)


                #plot area  roc and prc curve per model per settings
                # eval_plot.plot_roc_prc(prec, recall, fpr, tpr, title_suffix, plot_dir)

            avg_auprc = sum_auprc / float(number_of_runs)
            all_run_plot_dir = out_dir + 'Viz/'+ alg_spec_out_dir

            #box plot for auprc and auroc per cell line across all runs of a single model with single param setting
            # eval_plot.box_plot_auprc_auroc_all_runs(auprc_per_cell_line_all_runs, auroc_per_cell_line_all_runs, \
            #                                         title_suffix, 'sep', all_run_plot_dir)

            #box plot for auprc and auroc combining all cell lines across all runs of a single model with single param setting
            # eval_plot.box_plot_auprc_auroc_all_runs(auprc_all_runs, auroc_all_runs, \
            #                                         title_suffix, 'combo', all_run_plot_dir)


            if best_avg_auprc[alg] < avg_auprc:
                best_avg_auprc[alg] = avg_auprc

                # performance of the best model from each alg (at each run) is  kept here
                best_models_eval_score_dict[alg] = eval_scores
                best_model_param_dict[alg] = model_params[alg][i]

            i += 1

    #plots of best models from all algs
    plot_dir = out_dir+ 'Viz/' + 'all_alg/' + preprocess_params_str+'/'

    os.makedirs(plot_dir, exist_ok=True)
    eval_plot.plot_best_models_auprc_auroc_e_prec(early_prec_k, neg_fact, best_models_eval_score_dict, best_model_param_dict, plot_dir)


def find_best_param(should_run_algs, param_settings_dict, cross_val_type, kwargs, config_map):
    dataset_params, preprocess_params_str = get_data_preprocess_params(cross_val_type, config_map, kwargs)
    number_of_runs = config_map['ml_models_settings']['runs']
    n_folds = config_map['ml_models_settings']['cross_val']['folds']
    best_param={alg: str for alg in should_run_algs}
    min_loss = {alg: 10000 for alg in should_run_algs}

    for alg in should_run_algs:
        if alg in ['synverse','synverse_v2','synverse_v3','synverse_v4', 'synverse_tissuenet','synverse_nogenex','decagon','deepsynergy','dtf']:
            loss_across_all_settings_all_runs = {i:[] for i in range(len(param_settings_dict[alg]))} # the list will contain loss for 5 runs
            for run_ in range(number_of_runs):
                out_dir = config_map['project_dir'] + config_map['output_dir']+ alg + '/' + preprocess_params_str+'/run_'+str(run_)+'/'
                param_count=0
                for param in param_settings_dict[alg]:
                    model_param, model_info_str = create_model_info_string(alg, param)

                    #handle the slight inconsistency in saved deepsynergy loss file
                    if(alg =='deepsynergy')|(alg=='dtf'):
                        model_info_str = model_info_str[1:]

                    loss_file = out_dir + model_info_str + '_model_val_loss.txt'
                    file1 = open(loss_file, 'r')
                    print('loss file',loss_file)
                    lines = file1.readlines()

                    sum_loss = 0
                    count=0
                    for line in lines:
                        if 'val_loss' in line:
                            loss = line.split('val_loss: ')[1]
                            if loss == 'None\n':
                                loss=0
                            else:
                                loss = float(loss)
                            sum_loss = sum_loss+loss
                            count += 1

                    assert count == n_folds, 'problem in loss saving file'

                    avg_loss = sum_loss/float(n_folds)
                    loss_across_all_settings_all_runs[param_count].append(avg_loss)

                    param_count += 1

            for key in loss_across_all_settings_all_runs:
                avg_loss_across_mult_runs = np.mean(loss_across_all_settings_all_runs[key])
                if avg_loss_across_mult_runs<min_loss[alg]:
                    min_loss[alg] = avg_loss_across_mult_runs
                    best_param[alg] = param_settings_dict[alg][key]

            print('Best param for alg: ', alg, '\n', best_param[alg])
    return best_param

def plot_best_models(should_run_algs,param_settings_dict, cross_val_type, kwargs, config_map):
    best_param_settings_dict = find_best_param(should_run_algs, param_settings_dict, cross_val_type, kwargs, config_map)
    #save best params


    # get settings from config map
    dataset_params, preprocess_params_str = get_data_preprocess_params(cross_val_type, config_map, kwargs)
    number_of_runs = config_map['ml_models_settings']['runs']
    early_prec_k = kwargs.get('recall')
    neg_fact = config_map['ml_models_settings']['cross_val']['neg_fact']

    best_models_eval_score_dict = {alg: [] for alg in should_run_algs}


    for alg in should_run_algs:

        model_param, model_info_str = create_model_info_string(alg, best_param_settings_dict[alg])

        if alg!='dtf':
            pos_out_file = '/pos_val_scores' + model_info_str + '.tsv'
            neg_out_file = '/neg_val_scores' + model_info_str + '.tsv'
        else:
            pos_out_file = '/pos_scores' + model_info_str + '.tsv'
            neg_out_file = '/neg_scores' + model_info_str + '.tsv'

        eval_scores = []
        auprc_all_runs = {}
        auroc_all_runs={}

        for run_ in range(number_of_runs):
            out_dir = config_map['project_dir'] + config_map['output_dir']
            alg_spec_out_dir = alg + '/' + preprocess_params_str
            alg_spec_out_dir_per_run = alg_spec_out_dir + '/' + 'run_' + str(run_) + '/'
            alg_spec_out_path_per_run = out_dir + alg_spec_out_dir_per_run

            print('pos file name: ', alg_spec_out_dir_per_run + pos_out_file )


            if alg=='dtf':
                pos_df = pd.read_csv(alg_spec_out_path_per_run + pos_out_file)
                neg_df = pd.read_csv(alg_spec_out_path_per_run + neg_out_file)
                pos_df['predicted'] = pos_df['predicted'].apply(lambda x: re.split(r'[\[\]]', x)[1]).astype(np.float32)
                neg_df['predicted'] = neg_df['predicted'].apply(lambda x: re.split(r'[\[\]]', x)[1]).astype(np.float32)

            else:

                pos_df = pd.read_csv(alg_spec_out_path_per_run + pos_out_file, sep='\t')
                neg_df = pd.read_csv(alg_spec_out_path_per_run + neg_out_file, sep='\t')

            pos_neg_df = pd.concat([pos_df, neg_df], axis=0)\
                [['drug_1', 'drug_2', 'cell_line', 'predicted', 'true']]
            pos_neg_df = keep_one_from_symmetric_pairs(pos_neg_df, aggregate='max')


            prec, recall, fpr, tpr, auprc, auroc = compute.compute_roc_prc(pos_neg_df)
            auprc_all_runs[run_], auroc_all_runs[run_] = auprc, auroc

            e_prec_saving_file = alg_spec_out_path_per_run + 'early_prec' + model_info_str+ '.tsv'
            early_prec = compute.compute_early_prec(pos_neg_df, early_prec_k, e_prec_saving_file, True)

            eval_scores.append(EvalScore(auprc, auroc, early_prec))

            plot_dir = out_dir + 'Viz/'+ alg_spec_out_dir_per_run
            plot_dir = plot_dir

            os.makedirs(plot_dir, exist_ok=True)

        # performance of the best model from each alg (at each run) is  kept here
        best_models_eval_score_dict[alg] = eval_scores


    #plots of best models from all algs
    plot_dir = out_dir+ 'Viz/' + 'all_alg/' + preprocess_params_str+'/'

    os.makedirs(plot_dir, exist_ok=True)
    eval_plot.plot_best_models_auprc_auroc_e_prec(early_prec_k, neg_fact, best_models_eval_score_dict, plot_dir)

    f = open(plot_dir+'best_param_for_all_models.txt', 'w')
    for alg in best_param_settings_dict:
        f.write(alg + '\n')
        f.write(str(best_param_settings_dict[alg]) + '\n')
    f.close()



