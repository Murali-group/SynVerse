import pandas as pd
import numpy as np
import os
import pickle
import argparse
import yaml
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import copy
import time
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import logging
logging.basicConfig(filename='autoencoder_model.log', filemode='a', level=logging.DEBUG)

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
    group.add_argument('--config', type=str,
                       default="/home/tasnina/Projects/SynVerse/code/config-files/master_config_csbgpu3.yaml",
                       help="Configuration file for this script.")
    group.add_argument('--exfile', type=str,
        default="/home/tasnina/Projects/SynVerse/datasets/cosmic/CosmicCLP_CompleteGeneExpression.tsv",
        help=" name of the file containing z-score for expression level of genes in different cell lines")

    group.add_argument('--score', type=str,
                       default="Z_SCORE",
                       help="Z_SCORE or REGULATION. expression score to consider")

    group.add_argument('--preferred-model', type=list,
                       default=[4096, 1024, 256],
                       help="Layers for preferred model")

    group.add_argument('--force-cosmic-read', action = 'store_true', help = '''force_run ==True means read
                                            from the raw gene_expression file, not from the pkl file''')


    return parser

def plot_gene_expression(df, score_col, cell_line):

    plt.hist(list(df[score_col]))
    plt.xlabel(score_col)
    plt.ylabel('Number of genes')
    plt.title('gene expression in cell_line' + str(cell_line))
    plt.show()
    plt.close()


def create_cell_line_vs_gene_matrix(expression_df, filename, score_col, force_run=False):

    if (not os.path.exists(filename))| force_run :
        #in expression df, there is one sample id per cell line
        #convert the regulation values: (normal, under, over) into 3 numbers (0, 1, 2) respectively
        regulation_val_mapping = {'normal':0, 'under': 1, 'over': 2}
        if score_col == 'REGULATION':
            expression_df['REGULATION'] = expression_df['REGULATION'].apply(lambda x: regulation_val_mapping[x])

        all_genes = expression_df['GENE_NAME'].unique()
        final_df = pd.DataFrame(columns=all_genes)

        all_cell_line_ids = list(expression_df['SAMPLE_ID'].unique())
        # all_cell_line_ids = expression_df['SAMPLE_ID', 'SAMPLE_NAME'].unique()
        # cell_line_id_to_name = dict(zip(expression_df['SAMPLE_ID'], expression_df['SAMPLE_NAME']))

        print('cell line names: ', expression_df['SAMPLE_NAME'].unique())
        print(all_cell_line_ids)

        for cell_line_cosmic_id in all_cell_line_ids:
            df = expression_df[expression_df['SAMPLE_ID'] == cell_line_cosmic_id][['GENE_NAME', score_col]]
            # plot_gene_expression(df, score_col, cell_line)

            df = pd.DataFrame(df.values.T)
            df.columns = df.iloc[0]
            df.drop(0, axis = 0, inplace=True)

            # cell_line_name = cell_line_to_cosmic_id_mapping_df.loc[cell_line_cosmic_id]['cellName']
            # df.index = [cell_line_name]
            df.index = [cell_line_cosmic_id]

            print('cell line: ', cell_line_cosmic_id, 'genes: ', len(df.columns))
            final_df = pd.concat([final_df, df], axis=0)

        final_df.fillna(0, inplace=True)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(filename)

        final_df.to_csv(filename, sep='\t')


    final_df = pd.read_csv(filename, sep='\t')

    idx = final_df.index
    final_df = final_df.rename(columns={'Unnamed: 0': 'cosmic_id'})  # NURE
    return final_df



#ref https://github.com/yejinjkim/synergy-transfer/blob/master/Main.ipynb

class GeneExpressionDataset(Dataset):
    def __init__(self, gene_expression_df):
        self.gene_expression_df = gene_expression_df
    def __len__(self):
        return len(self.gene_expression_df)
    def get_num_features(self):
        return len(self.gene_expression_df.columns)
    def __getitem__(self, idx):
      return torch.tensor(self.gene_expression_df.iloc[idx]).to(torch.float32)


class GeneCompressingEncoder(nn.Module):
    def __init__(self, in_dim, h_sizes, out_dim, dropout):
        super(GeneCompressingEncoder, self).__init__()
        self.dropout = dropout

        self.all_layer_sizes = [in_dim] + h_sizes + [out_dim]
        self.all_layers = nn.ModuleList()

        for i in range(1, len(self.all_layer_sizes)):
            self.all_layers.append(nn.Linear(self.all_layer_sizes[i-1], self.all_layer_sizes[i]))


    def forward(self,x):
        out = F.dropout(F.relu(self.all_layers[0](x)), self.dropout, training=self.training)
        for i in range(1, len(self.all_layers)):
            out = F.dropout(F.relu(self.all_layers[i](out)), self.dropout, training=self.training)
        return out


class GeneCompressingDecoder(nn.Module):
    def __init__(self, in_dim, h_sizes, out_dim, dropout):
        super(GeneCompressingDecoder, self).__init__()
        self.dropout = dropout

        self.all_layer_sizes = [in_dim] + h_sizes + [out_dim]
        self.all_layers = nn.ModuleList()
        for i in range(1, len(self.all_layer_sizes)):
            self.all_layers.append(nn.Linear(self.all_layer_sizes[i - 1], self.all_layer_sizes[i]))

    def forward(self, x):
        out = F.dropout(F.relu(self.all_layers[0](x)), self.dropout, training=self.training)
        for i in range(1, len(self.all_layers)):
            out = F.dropout(F.relu(self.all_layers[i](out)), self.dropout, training=self.training)
        return out


class GeneCompressingAutoEncoder(nn.Module):
    def __init__(self, in_dim, h_sizes, out_dim, dropout=0.1):
        super(GeneCompressingAutoEncoder, self).__init__()
        self.encoder = GeneCompressingEncoder(in_dim, h_sizes, out_dim, dropout).to(dev)
        self.decoder = GeneCompressingDecoder(out_dim, list(np.flip(h_sizes)), in_dim, dropout).to(dev)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def val (gene_compressor_model, dataloader_val, criterion):
    gene_compressor_model.eval()
    total_loss = 0
    with torch.no_grad():
        for iteration, gene in enumerate(dataloader_val): #in each iteration a batch(e.g. size 256) of cell_lines are taken.
            gene = Variable(gene).float().to(dev)
            output = gene_compressor_model(gene)
            loss = criterion(output, gene)
            total_loss += loss.data
    return total_loss

def train_gene_compressing_model(dataset_train, dataset_val,\
                                 initial_num_genes, h_sizes, num_gene_compressed, epochs,\
                                 noise_weight=0.2, log_interval=1, patience=50):
    # model
    gene_compressor_model = GeneCompressingAutoEncoder(initial_num_genes, h_sizes,\
                                                       out_dim=num_gene_compressed, dropout=0.1).to(dev)

    #TODO: this loss function is applicable for Z_score only. use other loss function for 'REGULATION' value.
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(gene_compressor_model.parameters())

    n_epochs_with_val_loss_increment = 0
    prev_val_loss = 0
    min_val_loss = 10000000

    for epoch in range(1, epochs + 1):

        dataloader_train = DataLoader(dataset_train, batch_size=256, shuffle=True, num_workers=5)
        dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=True, num_workers=5)
        # train
        gene_compressor_model.train()
        total_loss = 0
        start_time = time.time()
        for iteration, gene in enumerate(dataloader_train): #in each iteration a batch(e.g. size 256) of cell_lines are taken.
            gene = Variable(gene).float()
            #add noise to the input
            noise = noise_weight * torch.randn(gene.shape)
            gene = gene.to(dev)
            noise = noise.to(dev)
            optimizer.zero_grad()
            output = gene_compressor_model(gene + noise)
            loss = criterion(output, gene)
            loss.backward()
            optimizer.step()
            total_loss += loss.data
            if iteration % log_interval == 0 and iteration > 0:
                cur_loss = total_loss.item() / log_interval
                # print(epoch, iteration, cur_loss)
                total_loss = 0

        val_loss = val(gene_compressor_model, dataloader_val, criterion)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_gene_compressor_model = copy.deepcopy(gene_compressor_model)
            n_epochs_with_val_loss_increment = 0

            print('changing best model at epoch: ', epoch)
        if val_loss >= min_val_loss:
            n_epochs_with_val_loss_increment+=1
            if n_epochs_with_val_loss_increment>patience:
                print('early stop: ', epoch)
                logging.info('Model:\n\n ')
                logging.info('Layers: '+ str(h_sizes+ [num_gene_compressed]))
                logging.info('Loss: ' + str(min_val_loss))
                logging.info('epochs: ' + str(epoch))
                return best_gene_compressor_model, min_val_loss
        print('epoch: ', epoch)

    logging.info('\n\nModel: ')
    logging.info('Layers: ' + str(h_sizes + [num_gene_compressed]))
    logging.info('Loss: ' + str(min_val_loss))
    logging.info('epochs: ' + str(epoch))

    return best_gene_compressor_model, min_val_loss

def handle_gene_compression_model_training(gene_expression_df_train, gene_expression_df_val, num_compressed_genes, \
                                           epochs, h_sizes, compressed_gene_expression_dir_path, force_run):

    model_file = compressed_gene_expression_dir_path+'/GeneCompressorModel_layers_' +\
                 str(h_sizes+[num_compressed_genes]) +'_e_'+ str(epochs) + '.p'

    #train data
    gene_expression_df_train_new = copy.deepcopy(gene_expression_df_train)
    gene_expression_df_train_new.drop('cosmic_id', axis = 1, inplace=True)
    gene_expression_dataset_train = GeneExpressionDataset(gene_expression_df_train_new)
    init_num_genes = gene_expression_dataset_train.get_num_features()

    #val data
    gene_expression_df_val_new = copy.deepcopy(gene_expression_df_val)
    gene_expression_df_val_new.drop('cosmic_id', axis = 1, inplace=True)
    gene_expression_dataset_val = GeneExpressionDataset(gene_expression_df_val_new)

    if (not os.path.exists(model_file)) | force_run:
        print('training model')
        #learn
        gene_compressing_model, val_loss = train_gene_compressing_model(gene_expression_dataset_train,\
                                    gene_expression_dataset_val, init_num_genes, \
                                    h_sizes, num_compressed_genes, epochs, noise_weight=0.01, log_interval=1)
        torch.save(gene_compressing_model.state_dict(), model_file)

    else:
        #load the saved model
        print('loading trained model')
        gene_compressing_model = GeneCompressingAutoEncoder(init_num_genes,h_sizes,
                                            out_dim=num_compressed_genes).to(dev)
        gene_compressing_model.load_state_dict(torch.load(model_file))
        dataloader_val = DataLoader(gene_expression_dataset_val, batch_size=64, shuffle=True, num_workers=5)
        val_loss = val(gene_compressing_model, dataloader_val, criterion = nn.MSELoss())

    return gene_compressing_model, val_loss


def load_model(init_num_genes, num_compressed_genes, epochs, h_sizes,
            compressed_gene_expression_dir_path):

    model_file = compressed_gene_expression_dir_path+'/GeneCompressorModel_layers_' +\
                 str(h_sizes+[num_compressed_genes]) +'_e_'+ str(epochs) + '.p'
    #load the saved model
    print('loading trained model')

    gene_compressing_model = GeneCompressingAutoEncoder(init_num_genes,h_sizes,
                                        out_dim=num_compressed_genes).to(dev)
    gene_compressing_model.load_state_dict(torch.load(model_file))

    return gene_compressing_model


def get_compressed_gene_expression(gene_compressing_model, gene_expression_for_common_cell_lines_df,\
                                   common_cosmic_id_to_cell_line_mapping_dict, compressed_gene_expression_dir_path, manual_or_auto):
    '''
    return: a dataframe containing cell_line_name as index. in each row it has the compresses gene expression of a cell line.
    '''
    gene_expression_for_common_cell_lines_df.set_index('cosmic_id', drop=True, inplace=True)
    # compressed_gene_expression_pkl_file = compressed_gene_expression_dir_path + 'compressed_gene_expression.pkl'
    compressed_gene_expression_file = compressed_gene_expression_dir_path + '/compressed_gene_expression_'+manual_or_auto + '.tsv'
    cosmic_ids_to_consider = list(gene_expression_for_common_cell_lines_df.index)

    gene_compressing_model.eval()
    with torch.no_grad():
        compressed_gene = np.array([gene_compressing_model.cpu().\
                                   encoder(torch.FloatTensor(gene_expression_for_common_cell_lines_df.iloc[d])).\
                                   data.numpy() for d in range(len(gene_expression_for_common_cell_lines_df))])

    df = pd.DataFrame(compressed_gene)
    df['cosmic_id'] = list(gene_expression_for_common_cell_lines_df.index)
    df['cell_line_name'] = df['cosmic_id'].apply(lambda x: common_cosmic_id_to_cell_line_mapping_dict[x])


    df.drop('cosmic_id', axis=1, inplace=True)
    df.to_csv(compressed_gene_expression_file, sep='\t')
    df = pd.read_csv(compressed_gene_expression_file, sep='\t', index_col=0)
    return df


def main(config_map, **kwargs):
    init_expression_file_path = kwargs.get('exfile')
    score_to_consider = kwargs.get('score')

    synergy_file_path = config_map['project_dir'] + config_map['inputs']['synergy']
    drug_comb_cell_line_file_path = config_map['project_dir'] + config_map['dataset_settings']['cell_lines']

    cell_line_vs_gene_matrix_file = config_map['project_dir'] + \
                                    config_map['inputs']['cell_lines'][score_to_consider]
    compressed_gene_expression_dir_path = os.path.dirname(cell_line_vs_gene_matrix_file)

    h_sizes_and_out_sizes_list = config_map['autoencoder_settings']['h_sizes_and_out']
    n_epochs = config_map['autoencoder_settings']['epochs']

    ############# COMBINING DRUGCOMB WITH DATA DOWNLOADED FROM COSMIC DATASET #################
    # extract gene expression data for cosmic ids in final_cosmic_ids
    expression_pkl_file = os.path.dirname(init_expression_file_path) + '/CosmicCLP_CompleteGeneExpression.pkl'
    if (not os.path.exists(expression_pkl_file))|kwargs.get('force_cosmic_read'):
        expression_df = pd.read_csv(init_expression_file_path, sep='\t')

        with open(expression_pkl_file, 'wb') as handle:
            pickle.dump(expression_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(expression_pkl_file, 'rb') as handle:
        expression_df = pickle.load(handle)


    cosmic_cell_lines = set(expression_df['SAMPLE_NAME'].unique())

    expression_df['SAMPLE_ID'] = expression_df['SAMPLE_ID'].astype(int)
    print(expression_df.nunique())

    #cell_line_vs_gene_expression_df has one column named 'cosmic_id'
    cell_line_vs_gene_expression_df = create_cell_line_vs_gene_matrix(expression_df,
                        cell_line_vs_gene_matrix_file, score_to_consider)
    print('number of cell lines for training: ', len(cell_line_vs_gene_expression_df))
    #train and find the best model
    min_val_loss = 1000000
    flag_train_model = False
    gene_expression_train_file = compressed_gene_expression_dir_path + '/gene_expression_train_file.tsv'
    gene_expression_val_file  = compressed_gene_expression_dir_path + '/gene_expression_val_file.tsv'
    if (not os.path.exists(gene_expression_train_file)) | (not os.path.exists(gene_expression_val_file)) | kwargs.get('force_new_train_val'):
        cell_line_vs_gene_expression_df_train = cell_line_vs_gene_expression_df.sample(frac=0.8, replace=False, random_state=1).reset_index(drop=True)
        cell_line_vs_gene_expression_df_val = cell_line_vs_gene_expression_df.drop(cell_line_vs_gene_expression_df_train.
                                                                                index, axis=0).reset_index(drop=True)

        cell_line_vs_gene_expression_df_train.to_csv(gene_expression_train_file, sep='\t')
        cell_line_vs_gene_expression_df_val.to_csv(gene_expression_val_file, sep='\t')

        flag_train_model = True

    cell_line_vs_gene_expression_df_train = pd.read_csv(gene_expression_train_file, sep='\t')
    cell_line_vs_gene_expression_df_val = pd.read_csv(gene_expression_val_file, sep='\t')
    ########## NOW in filter the compressed_gene_nparray such that cell_lines for which we have\
    # both SYNERGY VAL and COSMIC ID are present in drugcombdb are kept.

    synergy_df = pd.read_csv(synergy_file_path, sep='\t')
    cell_line_to_cosmic_id_mapping_df = pd.read_csv(drug_comb_cell_line_file_path,\
                                                    usecols=['cellName', 'cosmicId'])

    cell_line_to_cosmic_id_mapping_df.dropna(axis=0, inplace=True)
    cell_line_to_cosmic_id_mapping_df.drop_duplicates(inplace=True)
    cell_line_to_cosmic_id_mapping_df['cosmicId'] = cell_line_to_cosmic_id_mapping_df['cosmicId'].astype(int)
    cell_line_to_cosmic_id_mapping_df.set_index('cosmicId', inplace=True)

    cell_lines_with_synergy_val = set(synergy_df['Cell_line'])
    cell_lines_with_cosmic_id = set(cell_line_to_cosmic_id_mapping_df['cellName'])

    common_cell_lines = cell_lines_with_synergy_val.intersection(cell_lines_with_cosmic_id)

    common_cell_line_to_cosmic_id_mapping_df = cell_line_to_cosmic_id_mapping_df[
        cell_line_to_cosmic_id_mapping_df['cellName'].isin(common_cell_lines)]
    common_cosmic_id_to_cell_line_mapping_dict = dict(zip(common_cell_line_to_cosmic_id_mapping_df.index,\
                                                          common_cell_line_to_cosmic_id_mapping_df['cellName']))

    gene_expression_for_common_cell_lines_df = cell_line_vs_gene_expression_df[cell_line_vs_gene_expression_df['cosmic_id']\
                                            .isin(common_cosmic_id_to_cell_line_mapping_dict.keys())]

    if kwargs.get('preferred_model')==None:

        for h_sizes_and_out_size in h_sizes_and_out_sizes_list:
            print('current model : ', str(h_sizes_and_out_size))
            num_compressed_genes = h_sizes_and_out_size[-1]
            h_sizes = h_sizes_and_out_size[0:-1]
            gene_compressing_model, val_loss = handle_gene_compression_model_training(cell_line_vs_gene_expression_df_train,
                        cell_line_vs_gene_expression_df_val,  num_compressed_genes, n_epochs, h_sizes, \
                        compressed_gene_expression_dir_path, flag_train_model)

            if(val_loss<min_val_loss):
                best_gene_compressing_model = copy.deepcopy(gene_compressing_model)
                min_val_loss = val_loss
                layers_in_best_model = h_sizes_and_out_size

        print('layers for best model: ', layers_in_best_model)
        logging.info('layers for best model: '+ str(layers_in_best_model))

        compressed_gene_expression_in_common_cell_lines_df = \
            get_compressed_gene_expression(best_gene_compressing_model, gene_expression_for_common_cell_lines_df, \
                                           common_cosmic_id_to_cell_line_mapping_dict,
                                           compressed_gene_expression_dir_path, 'auto')

    else:
        #this is added so that I can choose a model manually to compress the gene_ecpression data.
        # AWhy do I need it? => I can decide to choose a model which might not be the best but the model can have
        # loss comparable to min_val_loss but with lower number of nodes in final layer.
        h_sizes_and_out_size = kwargs.get('preferred_model')
        print('Given preferred model: ', str(h_sizes_and_out_size))
        num_compressed_genes = h_sizes_and_out_size[-1]
        h_sizes = h_sizes_and_out_size[0:-1]

        init_num_genes = len(cell_line_vs_gene_expression_df_train.columns)-1
        preferred_gene_compressing_model = load_model(init_num_genes, num_compressed_genes, n_epochs, h_sizes,
            compressed_gene_expression_dir_path)


        compressed_gene_expression_in_common_cell_lines_df = \
            get_compressed_gene_expression(preferred_gene_compressing_model, gene_expression_for_common_cell_lines_df, \
                                           common_cosmic_id_to_cell_line_mapping_dict,
                                           compressed_gene_expression_dir_path,'manual')



if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)

