import evaluation.plot as metric_plot
import evaluation.compute as compute
import pandas as pd
import argparse
import yaml

def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        #config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.load(conf)
    # TODO check to make sure the inputs are correct in config_map

    return config_map, kwargs

def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="""Script to download and parse input files,
                                     and (TODO) run the  pipeline using them.""")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="/home/tasnina/Projects/SynVerse/code/config-files/master-config.yaml",
                       help="Configuration file for this script.")
    group.add_argument('--synergy', type=str, default="/synergy/synergy_labels.tsv",
                       help="Configuration file for this script.")
    group.add_argument('--drug_target', type=str, default="/drugs/drug_target_map.tsv",
                       help="Configuration file for this script.")

    group = parser.add_argument_group('FastSinkSource Pipeline Options')
    group.add_argument('--alg', '-A', dest='algs', type=str, action="append",
                       help="Algorithms for which to get results. Must be in the config file.\
                           If not specified, will get the list of algs with should_run set to True in the config file")
    group.add_argument('--force-run', action='store_true', default=False)

    return parser


def if_best_model(df, best_auprc, best_auroc):
    prec, recall, fpr, tpr, auprc, auc = compute.compute_roc_prc(df)

def evaluate(should_run_algs, config_map):

    #get settings from config map
    min_pairs_per_cell_line = config_map['synergy_data_settings']['min_pairs']
    max_pairs_per_cell_line = config_map['synergy_data_settings']['max_pairs']
    threshold = config_map['synergy_data_settings']['threshold']['val']
    neg_fact = config_map['ml_models_settings']['cross_val']['neg_fact']
    number_of_runs = config_map['ml_models_settings']['cross_val']['runs']

    preds_from_best_models = {alg: pd.DataFrame() for alg in should_run_algs}

    best_avg_auprc = {alg: 0 for alg in should_run_algs}
    best_avg_auroc = {alg: 0 for alg in should_run_algs}


    outputs_df = {alg: [] for alg in should_run_algs}
    for alg in should_run_algs:
        if alg == 'decagon':
            decagon_settings = config_map['ml_models_settings']['algs']['decagon']
            lr = decagon_settings['learning_rate']
            epochs = decagon_settings['epochs']
            batch_size = decagon_settings['batch_size']
            dr = decagon_settings['dropout']
            use_drug_feat_options = decagon_settings['use_drug_feat']
            for drug_feat_option in use_drug_feat_options:
                sum_auprc = 0
                sum_auroc = 0
                for run_ in range(number_of_runs):
                    out_dir = config_map['project_dir'] + config_map['output_dir'] + alg + '/' + 'pairs_' + str(min_pairs_per_cell_line) +\
                              '_' + str(max_pairs_per_cell_line) + '_th_' + str(threshold) + '_' + 'neg_' + str(neg_fact) + '/'

                    pos_out_file = out_dir + 'run_' + str(run_) + '/' + '/pos_val_scores' + '_drugfeat_' + str(drug_feat_option)+ \
                                   '_e_' + str(epochs) + '_lr_' + str(lr) + '_batch_' + str(batch_size) + '_dr_' + str(dr) + '.tsv'
                    neg_out_file = out_dir + 'run_' + str(run_) + '/' + '/neg_val_scores' + '_drugfeat_' + str(drug_feat_option) + \
                                   '_e_' + str(epochs) + '_lr_' + str(lr) + '_batch_' + str(batch_size) + '_dr_' + str(dr) + '.tsv'
                    pos_df = pd.read_csv(pos_out_file, sep='\t')
                    neg_df = pd.read_csv(neg_out_file, sep='\t')
                    pos_neg_df = pd.concat([pos_df, neg_df], axis=0) \
                        [['drug_1', 'drug_2', 'cell_line', 'model_score', 'predicted', 'true']]
                    outputs_df[alg].append(pos_neg_df)


                    print('plot: ')
                    title_suffix = 'run_' + str(run_) + '_pairs_' + str(min_pairs_per_cell_line) + '_' + str(
                        max_pairs_per_cell_line) + \
                                   '_drugfeat_' + str(drug_feat_option) + \
                                   '_e_' + str(epochs) + '_lr_' + str(lr) + '_batch_' + str(batch_size) + '_dr_' + str(dr)
                    plot_dir = out_dir + 'plot/'
                    metric_plot.plot_predicted_score_distribution(pos_neg_df, title_suffix, plot_dir)
                    metric_plot.scatter_plot_auprc_auroc(pos_neg_df, title_suffix, plot_dir)

                    prec, recall, fpr, tpr, auprc, auroc = compute.compute_roc_prc(pos_neg_df)
                    sum_auprc += auprc
                    sum_auroc += auroc
                avg_auprc = sum_auprc/float(number_of_runs)
                avg_auroc = sum_auroc / float(number_of_runs)

                if best_avg_auprc[alg] < avg_auprc:
                    best_avg_auprc[alg] = avg_auprc
                    preds_from_best_models[alg] = pos_neg_df

                # metric_plot.performance_metric_evaluation_per_alg(outputs_df[alg], alg, config_map)

        elif alg == 'synverse':
            synverse_settings = config_map['ml_models_settings']['algs']['synverse']
            lr = synverse_settings['learning_rate']
            epochs = synverse_settings['epochs']
            batch_size = synverse_settings['batch_size']
            dr = synverse_settings['dropout']
            h_sizes = synverse_settings['h_sizes']
            use_drug_feat_options = synverse_settings['use_drug_feat']

            for drug_feat_option in use_drug_feat_options:
                for run_ in range(number_of_runs):
                    out_dir = config_map['project_dir'] + config_map['output_dir'] + alg + '/' + \
                              'pairs_' + str(min_pairs_per_cell_line) + '_' + str(
                        max_pairs_per_cell_line) + '_th_' + str(
                        threshold) + '_' + 'neg_' + str(neg_fact) + '/'
                    pos_out_file = out_dir + 'run_' + str(run_) + '/' + '/pos_val_scores' + '_hsize_' + str(h_sizes) + '_drugfeat_' + str(
                        drug_feat_option) + '_e_' + str(epochs) + '_lr_' + str(lr) + '_batch_' + str(batch_size) + '_dr_' + str(dr) + '.tsv'
                    neg_out_file = out_dir + 'run_' + str(run_) + '/' + '/neg_val_scores' + '_hsize_' + str( h_sizes) + '_drugfeat_' + str(
                        drug_feat_option) + '_e_' + str(epochs) + '_lr_' + str(lr) + '_batch_' + str(batch_size) + '_dr_' + str(dr) + '.tsv'

                    pos_df = pd.read_csv(pos_out_file, sep='\t')
                    neg_df = pd.read_csv(neg_out_file, sep='\t')
                    pos_neg_df = pd.concat([pos_df, neg_df], axis=0) \
                        [['drug_1', 'drug_2', 'cell_line', 'predicted', 'true']]
                    outputs_df[alg].append(pos_neg_df)


                    print('plot: ')
                    title_suffix = 'run_' + str(run_) + '_pairs_' + str(min_pairs_per_cell_line) + '_' + str(
                        max_pairs_per_cell_line) + \
                                   '_drugfeat_' + str(drug_feat_option) + \
                                   '_e_' + str(epochs) + '_lr_' + str(lr) + '_batch_' + str(batch_size) + '_dr_' + str(
                        dr)
                    plot_dir = out_dir + 'plot/'
                    metric_plot.plot_predicted_score_distribution(pos_neg_df, title_suffix, plot_dir)
                    metric_plot.scatter_plot_auprc_auroc(pos_neg_df, title_suffix, plot_dir)
                # metric_plot.performance_metric_evaluation_per_alg(outputs_df[alg], alg, config_map)

        elif alg == 'deepsynergy':

            ds_settings = config_map['ml_models_settings']['algs']['deepsynergy']
            epochs = ds_settings['epochs']
            batch_size = ds_settings['batch_size']
            act_func = ds_settings['act_func']

            layer_setups = ds_settings['layers']
            lrs = ds_settings['learning_rate']
            in_hid_dropouts = ds_settings['in_hid_dropouts']
            # use_drug_feat_options = decagon_settings['use_drug_feat']
            # for drug_feat_option in use_drug_feat_options:
            for layer_setup in layer_setups:
                for lr in lrs:
                    for in_hid_dropout in in_hid_dropouts:
                        for run_ in range(number_of_runs):
                            out_dir = config_map['project_dir'] + config_map['output_dir'] + alg + '/' + \
                                      'pairs_' + str(min_pairs_per_cell_line) + '_' + str(
                                max_pairs_per_cell_line) + '_th_' + str(
                                threshold) + '_' + 'neg_' + str(neg_fact) + '/'

                            input_dropout = in_hid_dropout[0]
                            dropout = in_hid_dropout[1]

                            pos_out_file = out_dir + 'run_' + str(run_) + '/' + '/pos_val_scores' + '_layers_' + str(layer_setup) + '_e_'+\
                                           str(epochs) + '_lr_' + str(lr) + '_batch_' + str(batch_size) + '_dr_' + \
                                           str(input_dropout) + '_' + str(dropout) + '_act_' + str(act_func) + '.tsv'

                            neg_out_file = out_dir + 'run_' + str(run_) + '/' + '/neg_val_scores' + '_layers_' + str(layer_setup) + '_e_'+ \
                                           str(epochs) + '_lr_' + str(lr) + '_batch_' + str(batch_size) + '_dr_' + \
                                           str(input_dropout) + '_' + str(dropout) + '_act_' + str(act_func) + '.tsv'
                            pos_df = pd.read_csv(pos_out_file, sep='\t')
                            neg_df = pd.read_csv(neg_out_file, sep='\t')
                            pos_neg_df = pd.concat([pos_df, neg_df], axis=0) \
                                [['drug_1', 'drug_2', 'cell_line', 'predicted', 'true']]
                            outputs_df[alg].append(pos_neg_df)


                            print('plot: ')
                            title_suffix = 'run_' + str(run_) + '_pairs_' + str(min_pairs_per_cell_line) + '_' + str(
                                max_pairs_per_cell_line) + \
                                           '_layers_' + str(layer_setup) + '_e_' + str(epochs) + '_lr_' + str(
                                lr) + '_batch_' + str(batch_size) + '_dr_' + \
                                           str(input_dropout) + '_' + str(dropout)
                            plot_dir = out_dir + 'plot/'

                            metric_plot.plot_predicted_score_distribution(pos_neg_df, title_suffix, plot_dir)
                            metric_plot.scatter_plot_auprc_auroc(pos_neg_df, title_suffix, plot_dir)
                metric_plot.performance_metric_evaluation_per_alg(outputs_df[alg], alg, config_map)



config_map, kwargs = parse_args()
evaluate(['synverse', 'deepsynergy'], config_map)