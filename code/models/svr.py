import os
import pandas as pd
import numpy as np
import argparse
import yaml
import logging
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    return  kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to Support vector regression. ")

    parser.add_argument("--input-dir", type=str,
                           help="input prefix where training input file will be placed. " )
    parser.add_argument("--output-dir", type=str,
                       help="Output prefix where final output file will be placed. " )

    return parser
def pearson_correlation_func(y_true, y_pred):
    pearsons_r, p_value = stats.pearsonr(y_true, y_pred)
    return pearsons_r
def plot_score_for_diff_c_val(score_dict, score_type, kernel, cell_line, output_dir):
    x = list(score_dict.keys())
    y = list(score_dict.values())
    plt.plot(x, y)
    plt.xlabel('C')
    plt.ylabel(score_type)

    title = score_type + ' vs C for '+ kernel + ' kernel in '+ cell_line
    plt.title(title)
#     plt.figure(figsize=[8, 6])

    file_name = output_dir + cell_line + '/' +  kernel + '_' + score_type + '_plot.pdf'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    plt.savefig(file_name, bbox_inches ="tight", format = 'pdf')
    print(file_name, ' saved')

    # plt.show()
    plt.close()

def main(**kwargs):
    # input_train_dir = "/data/tasnina/drug-synergy-prediction/inputs/train/"
    # output_dir = "/data/tasnina/drug-synergy-prediction/outputs/"
    log_dir = '/home/tasnina/Projects/drug-synergy-prediction/code/log/'
    log_file= log_dir+'logger_SVR.log'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, filemode='w',level = logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

    input_train_dir =  kwargs.get('input_dir')
    output_dir =kwargs.get('output_dir')

    list_of_cell_lines = ['NCI-H226', 'HCC-2998', 'MDA-MB-435', 'DIPG25', 'COLO 205', 'HCT-15', 'HOP-92', 'TK-10', 'U251', 'HL-60(TB)', 'CCRF-CEM', 'EKVX',
    'NCI-H522', 'SW-620','MALME-3M', 'RPMI-8226','NCI-H322M','UO-31','MDA-MB-231', 'SK-MEL-5','HS 578T',
    'M14', 'T-47D', 'SR', 'SF-539', 'OVCAR-8','OVCAR-5', 'SK-OV-3', 'OVCAR-4', 'TMD8',
    'HOP-62', 'NCIH23', 'IGROV1', 'ACHN', 'MCF7','SNB-19', '786-0', 'MDA-MB-468', 'K-562', 'SF-268', 'KBM-7',
    'KM12', 'SF-295','A498', 'HCT116', 'OVCAR3', 'CAKI-1', 'BT-549', 'UACC62',
    'A549','NCI-H460','SN12C', 'MOLT-4', 'SNB-75', 'HT29', 'RXF 393', 'LOX IMVI', 'UACC-257', 'SK-MEL-28', 'DU-145']

    scoring = {'r2': make_scorer(r2_score),'mean_squared_error':make_scorer(mean_squared_error),'pearsons_correlation_score':make_scorer(pearson_correlation_func)}


    for cell_line in list_of_cell_lines:
        feature_file = input_train_dir+cell_line+'_features.tsv'
        label_file = input_train_dir+cell_line+'_labels.tsv'

        feature = pd.read_csv(feature_file,sep='\t').to_numpy()
        # print(feature, feature.shape)
        label = pd.read_csv(label_file,sep='\t')['Loewe'].to_numpy()
        # print(label, label.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            feature, label, test_size=0.1, random_state=0)

        for k in ['poly', 'rbf', 'linear']:
            file_name = output_dir + cell_line + '/' +  k + '_r2_mse_pearsons_score.tsv'

        # try:
            if(not os.path.exists(file_name)):
                os.makedirs(os.path.dirname(file_name), exist_ok=True)

                avg_r2_dict = {}
                avg_mse_dict = {}
                avg_pearsons_dict = {}
                for c in range(1,100,5):
                    SVR_model = SVR(C=c, epsilon=0.2, kernel = k)
                    scores = cross_validate(SVR_model, X_train,y_train, cv=5, scoring = scoring)
                    avg_r2 = np.average(scores['test_r2'])
                    avg_mse =  np.average(scores['test_mean_squared_error'])
                    avg_pearsons = np.average(scores['test_pearsons_correlation_score'])

                    avg_r2_dict[c] = avg_r2
                    avg_mse_dict[c] = avg_mse
                    avg_pearsons_dict[c] = avg_pearsons

                    print(avg_r2)
                    print(avg_mse)
                    print(avg_pearsons)
                #save these dictionary values as a df in file
                r2_mse_pearsons_score_df = pd.concat([pd.Series(avg_r2_dict,name='r2'), \
                                              pd.Series(avg_mse_dict,name = 'mse'),\
                                              pd.Series(avg_pearsons_dict,name = 'pearsons')],axis=1)


                r2_mse_pearsons_score_df.to_csv(file_name, sep='\t')

            else:
                print(file_name + ' already exists')

            r2_mse_pearsons_score_df = pd.read_csv(file_name, sep='\t', index_col = 'Unnamed: 0')
    #         print(r2_rmse_score_df.head())
            avg_r2_dict = r2_mse_pearsons_score_df['r2'].to_dict()
    #         print(avg_r2_dict.keys())
            avg_mse_dict = r2_mse_pearsons_score_df['mse'].to_dict()
            avg_pearsons_dict = r2_mse_pearsons_score_df['pearsons'].to_dict()
            #plot and save rmse, r2 scores

            plot_score_for_diff_c_val(avg_r2_dict, 'r2', k, cell_line, output_dir)
            plot_score_for_diff_c_val(avg_mse_dict,'mse', k, cell_line, output_dir)
            plot_score_for_diff_c_val(avg_pearsons_dict,'pearsons', k, cell_line, output_dir)


        # except:
        #     print('could not find svr result for %s'%(file_name))
        #     logging.info("could not find svr result %s",file_name )

# if __name__ == "__main__":
#     main()
if __name__ == "__main__":
    kwargs = parse_args()
    main( **kwargs)
