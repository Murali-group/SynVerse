import os

import evaluation.compute as compute
import pandas as pd
import argparse
import yaml
from evaluation.utils import EvalScore, set_title_suffix, keep_one_from_symmetric_pairs
import evaluation.plot as eval_plot

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
                       default="/home/tasnina/Projects/SynVerse/code/config-files/master-config.yaml",
                       help="Configuration file for this script.")

    group.add_argument('--recall', type=float,
                       default=0.3,
                       help="recall value for early precision")

    return parser




def get_performance_scores(pos_neg_df, early_prec_recall_val):
    prec, recall, fpr, tpr, auprc, auroc = compute.compute_roc_prc(pos_neg_df)
    early_prec = compute.compute_early_prec(pos_neg_df, early_prec_recall_val)
    return auprc, auroc, early_prec

def evaluate(should_run_algs, cross_val_type, kwargs, config_map):
    # get settings from config map
    min_pairs_per_cell_line = config_map['synergy_data_settings']['min_pairs']
    max_pairs_per_cell_line = config_map['synergy_data_settings']['max_pairs']
    threshold = config_map['synergy_data_settings']['threshold']['val']
    neg_fact = config_map['ml_models_settings']['cross_val']['neg_fact']
    number_of_runs = config_map['ml_models_settings']['cross_val']['runs']

    early_prec_k = kwargs.get('recall')
    # preds_from_best_models = {alg: pd.DataFrame() for alg in should_run_algs}
    best_avg_auprc = {alg: 0 for alg in should_run_algs}
    best_avg_auroc = {alg: 0 for alg in should_run_algs}

    # predictions_df = {alg: [] for alg in should_run_algs}


    model_params = {alg: [] for alg in should_run_algs} #dictionary of  list of dictionaries. settings for each alg will be kept in the innermost dictionary
    pos_neg_file_names = {alg: [] for alg in should_run_algs} #dictionary of list of tuples. each tupe contain two strings\
                                                              # for positive and negative file names


    #get prediction file names for each model and settings
    for alg in should_run_algs:
        if alg == 'decagon':
            decagon_settings = config_map['ml_models_settings']['algs']['decagon']
            lr = decagon_settings['learning_rate']
            epochs = decagon_settings['epochs']
            batch_size = decagon_settings['batch_size']
            dr = decagon_settings['dropout']
            use_drug_feat_options = decagon_settings['use_drug_feat']
            for drug_feat_option in use_drug_feat_options:
                #
                # model_param = {'min_pairs': min_pairs_per_cell_line,
                #                'max_pairs': max_pairs_per_cell_line, 'neg': neg_fact, 'th':  threshold,  \
                #                'lr': lr, 'e': epochs, 'batch': batch_size, 'dr': dr, \
                #                'd_feat': drug_feat_option}


                model_param = {'min_pairs': min_pairs_per_cell_line,
                               'max_pairs': max_pairs_per_cell_line, 'neg': neg_fact, 'th':  threshold,  \
                               'lr': lr, 'e': epochs, 'batch': batch_size, 'dr': dr, \
                               'd_feat': drug_feat_option}

                model_params[alg].append(model_param)

                pos_out_file =  '/pos_val_scores' + '_drugfeat_' + str( drug_feat_option) + '_e_' + str(epochs) +\
                                 '_lr_' + str(lr) + '_batch_' + str(batch_size) + '_dr_' + str(dr) + '.tsv'


                neg_out_file = '/neg_val_scores' + '_drugfeat_' + str(drug_feat_option) + \
                                '_e_' + str(epochs) + '_lr_' + str(lr) + '_batch_' + str(batch_size) + '_dr_' + str(dr) + '.tsv'
                # model output files
                pos_neg_file_names[alg].append((pos_out_file, neg_out_file))


        elif alg == 'synverse':
            synverse_settings = config_map['ml_models_settings']['algs']['synverse']
            lr = synverse_settings['learning_rate']
            epochs = synverse_settings['epochs']
            batch_size = synverse_settings['batch_size']
            dr = synverse_settings['dropout']
            h_sizes = synverse_settings['h_sizes']
            use_drug_feat_options = synverse_settings['use_drug_feat']

            for drug_feat_option in use_drug_feat_options:
                # model parameter and model output files
                model_param = {'min_pairs': min_pairs_per_cell_line, \
                               'max_pairs': max_pairs_per_cell_line,  'neg': neg_fact,'th':  threshold, 'h_sizes': h_sizes, \
                               'lr': lr, 'e': epochs, 'batch': batch_size, 'dr': dr, \
                               'd_feat': drug_feat_option}

                # model_param = {'h_sizes': h_sizes,
                #                'lr': lr,
                #                'e': epochs,
                #                'dr': dr}
                model_params[alg].append(model_param)

                pos_out_file = '/pos_val_scores' + '_hsize_' + str(h_sizes) + '_drugfeat_' + str(drug_feat_option) +\
                            '_e_' + str(epochs) + '_lr_' + str(lr) + '_batch_' + str(batch_size) + '_dr_' + str(dr) + '.tsv'
                neg_out_file = '/neg_val_scores' + '_hsize_' + str(h_sizes) +'_drugfeat_' + str(drug_feat_option) +\
                            '_e_' + str(epochs) + '_lr_' + str(lr) + '_batch_' + str(batch_size) + '_dr_' + str(dr) + '.tsv'
                pos_neg_file_names[alg].append((pos_out_file, neg_out_file))


        elif alg == 'deepsynergy':
            ds_settings = config_map['ml_models_settings']['algs']['deepsynergy']
            lrs = ds_settings['learning_rate']
            epochs = ds_settings['epochs']
            batch_size = ds_settings['batch_size']
            in_hid_dropouts = ds_settings['in_hid_dropouts']
            layer_setups = ds_settings['layers']
            act_func = ds_settings['act_func']

            for layer_setup in layer_setups:
                for lr in lrs:
                    for in_hid_dropout in in_hid_dropouts:
                        input_dropout = in_hid_dropout[0]
                        dropout = in_hid_dropout[1]

                        model_param = {'min_pairs': min_pairs_per_cell_line,
                                       'max_pairs': max_pairs_per_cell_line,
                                       'neg': neg_fact, 'th': threshold,
                                       'layers': layer_setup, 'lr': lr, 'e': epochs,
                                       'batch': batch_size, 'dr': in_hid_dropout,
                                       'act_': act_func}
                        # model_param = {'layers': layer_setup, 'lr': lr, 'e': epochs,
                        #                'batch': batch_size, 'dr': in_hid_dropout,
                        #                'act_': act_func}

                        model_params[alg].append(model_param)

                        pos_out_file = '/pos_val_scores' + '_layers_' + str(layer_setup) + '_e_' + str(epochs) + '_lr_'+\
                            str(lr) + '_batch_' + str(batch_size) + '_dr_' + \
                            str(input_dropout) + '_' + str(dropout) + '_act_' + str(act_func) + '.tsv'

                        neg_out_file = '/neg_val_scores' + '_layers_' + str(layer_setup) + '_e_' + str(epochs) + '_lr_'+\
                            str(lr) + '_batch_' + str(batch_size) + '_dr_'+ str(input_dropout) + '_' + str(dropout) + '_act_'+\
                            str(act_func) + '.tsv'

                        # model parameter and model output files
                        pos_neg_file_names[alg].append((pos_out_file, neg_out_file))


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
            for run_ in range(number_of_runs):
                # out_dir = config_map['project_dir'] + config_map['output_dir'] + alg + '/' + 'pairs_' + \
                #     str(min_pairs_per_cell_line) + '_' + str(max_pairs_per_cell_line) + '_th_' + str(threshold) +\
                #     '_' + 'neg_' + str(neg_fact) + '/'+'run_' + str(run_) + '/'

                out_dir = config_map['project_dir'] + config_map['output_dir'] + alg + '/' + cross_val_type + '/' + \
                          'pairs_' + str(min_pairs_per_cell_line) + '_' + str(max_pairs_per_cell_line) + '_th_' + str(
                    threshold) +'_' + 'neg_' + str(neg_fact) + '_' + kwargs.get('cvdir') + '/'+'run_' + str(run_) + '/'

                pos_df = pd.read_csv(out_dir + pos_out_file, sep='\t')
                neg_df = pd.read_csv(out_dir + neg_out_file, sep='\t')
                pos_neg_df = pd.concat([pos_df, neg_df], axis=0)\
                    [['drug_1', 'drug_2', 'cell_line', 'predicted', 'true']]
                pos_neg_df = keep_one_from_symmetric_pairs(pos_neg_df, aggregate='max')



                prec, recall, fpr, tpr, auprc, auroc = compute.compute_roc_prc(pos_neg_df)

                e_prec_saving_file = out_dir + '.'.join(pos_out_file.split('.')[0:-1]). \
                    replace('pos_val_scores', '') + '.tsv'
                early_prec = compute.compute_early_prec(pos_neg_df, early_prec_k, e_prec_saving_file, force_run=False)

                eval_scores.append(EvalScore(auprc, auroc, early_prec))
                sum_auprc += auprc

                print('plot: ')
                # plot_dir = config_map['project_dir'] + config_map['output_dir'] + 'Viz/' + alg + '/' + 'pairs_' + \
                #     str(min_pairs_per_cell_line) + '_' + str(max_pairs_per_cell_line) + '_th_' + str(threshold) +\
                #     '_' + 'neg_' + str(neg_fact) + '/'+'run_' + str(run_) + '/'

                plot_dir = config_map['project_dir'] + config_map['output_dir'] + 'Viz/' + alg + '/' + cross_val_type +\
                           '/pairs_'+ str(min_pairs_per_cell_line) + '_' + str(max_pairs_per_cell_line) + '_th_' +\
                           str(threshold)+'_' + 'neg_' + str(neg_fact) + '_' + kwargs.get('cvdir') + '/' + 'run_' +\
                           str(run_) + '/'
                os.makedirs(plot_dir, exist_ok=True)

                title_suffix = alg+'_run_' + str(run_) + set_title_suffix(model_params[alg][i])
                eval_plot.plot_predicted_score_distribution(pos_neg_df, title_suffix, plot_dir)
                eval_plot.scatter_plot_auprc_auroc(pos_neg_df, title_suffix, plot_dir)
                eval_plot.plot_roc_prc(prec, recall, fpr, tpr, title_suffix, plot_dir)

            avg_auprc = sum_auprc / float(number_of_runs)

            if best_avg_auprc[alg] < avg_auprc:
                best_avg_auprc[alg] = avg_auprc

                # performance of the best model from each alg (at each run) is  kept here
                best_models_eval_score_dict[alg] = eval_scores
                best_model_param_dict[alg] = model_params[alg][i]

            i += 1

    #plots of best models from all algs
    plot_dir = config_map['project_dir'] + config_map['output_dir'] + 'Viz/'+'all_alg/' + cross_val_type + '/pairs_' + \
                    str(min_pairs_per_cell_line) + '_' + str(max_pairs_per_cell_line) + '_th_' + str(threshold) +\
                    '_' + 'neg_' + str(neg_fact)+'_' + kwargs.get('cvdir') + '/'

    os.makedirs(plot_dir, exist_ok=True)
    eval_plot.plot_best_models_auprc_auroc_e_prec(early_prec_k, neg_fact, best_models_eval_score_dict, best_model_param_dict, plot_dir)

