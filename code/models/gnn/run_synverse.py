from __future__ import division
from __future__ import print_function

import time
import os, psutil

import gc
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv, RGCNConv
from torch_geometric.nn.inits import reset

import models.gnn.gnn_utils as utils
from models.gnn.input_output_preparation import *
import models.utils as model_utils
import models.gnn.cross_validation as cross_val
from models.gnn.gnn_minibatch import MinibatchHandler

from models.gnn.decoder_modules import NNDecoder,NNDecoder_nogenex, DedicomDecoder, BilinearDecoder
from models.gnn.encoder_modules import  Encoder
import wandb

import logging
logging.basicConfig(filename='models/gnn/synverse_model.log', filemode='a', level=logging.DEBUG)
# wandb.login()


EPS = 1e-15
MAX_LOGVAR = 10
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
process = psutil.Process(os.getpid())




#
# def sigmoid(x):
#     return 1. / (1 + np.exp(-x))

class SynverseModel(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder, decoders):
        super(SynverseModel, self).__init__()
        self.encoder = encoder
        self.decoders = decoders
        SynverseModel.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        for edge_type in self.decoders:
            reset(self.decoders[edge_type])

    # def encode(self, *args, **kwargs):
    #     r"""Runs the encoder and computes node-wise latent variables."""
    #     return self.encoder(*args, **kwargs)

    def encode(self):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder()

    # def decode(self, *args, **kwargs):
    #     r"""Runs the decoder and computes edge probabilties."""
    #     return self.decoder(*args, **kwargs)

    def recon_loss(self, z, batch_pos_edge_index, batch_neg_edge_index, edge_type, edge_sub_type_idx):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """
        pos_loss = -torch.log(self.decoders[edge_type](z, batch_pos_edge_index,
                    edge_sub_type_idx, is_sigmoid=True) + EPS).mean()

        neg_loss = -torch.log(1 - self.decoders[edge_type](z, batch_neg_edge_index,
                    edge_sub_type_idx, is_sigmoid=True) + EPS).mean()

        # return pos_loss
        return pos_loss + neg_loss

    def predict(self,  z, batch_pos_edge_index, batch_neg_edge_index, edge_type, edge_sub_type_idx):

        pos_pred = self.decoders[edge_type](z, batch_pos_edge_index, edge_sub_type_idx, is_sigmoid=True)
        neg_pred = self.decoders[edge_type](z, batch_neg_edge_index, edge_sub_type_idx, is_sigmoid=True)

        pos_loss = -torch.log(pos_pred + EPS).mean()
        neg_loss = -torch.log(1 - neg_pred + EPS).mean()

        loss = pos_loss + neg_loss


        pos_y = torch.ones(batch_pos_edge_index.size(1)).to(dev)
        neg_y = torch.zeros(batch_neg_edge_index.size(1)).to(dev)

        pos_pred = torch.stack([batch_pos_edge_index[0], batch_pos_edge_index[1], pos_pred, pos_y], dim=0)
        neg_pred = torch.stack([batch_neg_edge_index[0], batch_neg_edge_index[1], neg_pred, neg_y], dim=0)

        return pos_pred, neg_pred, loss


def train(model, optimizer,  batch_pos_train_edges, batch_neg_train_edges, edge_type, edge_sub_type_idx ):

    model.train()
    optimizer.zero_grad()
    z = model.encode()

    #normalize z and gene expression data here
    loss = model.recon_loss(z, batch_pos_train_edges, batch_neg_train_edges, edge_type, edge_sub_type_idx)
    loss.backward()
    optimizer.step()

    return float(loss)


def val(model, pos_edge_index, neg_edge_index, edge_type, edge_sub_type_idx):
    model.eval()

    with torch.no_grad():
        z = model.encode()

    #normalize z and gene expression data here
    pos_pred, neg_pred, val_loss = model.predict(z, pos_edge_index, neg_edge_index, edge_type, edge_sub_type_idx)
    return pos_pred, neg_pred, val_loss

def train_log(loss, wandb_step, edge_type, edge_name):
    loss = float(loss)
    edge_type_idx = utils.edge_type_to_idx(edge_type)
    l = 'batch_wise_loss_'+ edge_name
    wandb.log({l: loss, 'edge_type':edge_type_idx },\
              step = wandb_step)
    # wandb.log({'Epoch': epoch, 'loss': loss}, step=wandb_step)



def compute_loss(model, edge_type, pos_edges_split_dict, neg_edges_split_dict,
                          edge_type_wise_number_of_subtypes):
    total_loss = 0
    n_cell_lines = 0
    for edge_sub_type in range(edge_type_wise_number_of_subtypes[edge_type]):
        cell_line_wise_loss = 0
        is_empty_cell_line=False
        for split_idx in range(len(pos_edges_split_dict[edge_type][edge_sub_type])):
            pos_edges = pos_edges_split_dict[edge_type][edge_sub_type][split_idx].to(dev)
            neg_edges = neg_edges_split_dict[edge_type][edge_sub_type][split_idx].to(dev)
            if pos_edges.size()[1]!=0:
                batch_wise_pos_pred, batch_wise_neg_pred, loss = val(model, pos_edges,
                                                                           neg_edges, edge_type,
                                                                           edge_sub_type)
                cell_line_wise_loss += loss
            else:
                is_empty_cell_line = True
                break
        if not is_empty_cell_line:
            cell_line_wise_loss = cell_line_wise_loss/float(len(pos_edges_split_dict[edge_type][edge_sub_type]))
            total_loss += cell_line_wise_loss.to('cpu').detach().item()
            n_cell_lines+=1
    if n_cell_lines>0:
        avg_loss =  total_loss/float(n_cell_lines)
    else:
        avg_loss = None
    return avg_loss



def plot_loss(avg_loss, edge_type, train_or_val, wandb_step):
    wandb.log({train_or_val + '_loss_' + edge_type: avg_loss}, step=wandb_step)


def run_synverse_model(alg, ppi_sparse_matrix, gene_2_idx, drug_target_df, drug_maccs_keys_feature_df,
                        synergy_df, non_synergy_df,
                       cell_2_idx, idx_2_cell,
                       folds_pos_drug_drug_edges, folds_neg_drug_drug_edges,
                       cross_val_dir, neg_sampling_type,
                       out_dir, params,
                       config_map,  gene_expression_feature_df = None, use_drug_based_batch_end=False, do_save_best_model = False):

    edge_types = ['gene_gene', 'target_drug', 'drug_target', 'drug_drug']
    #model setup
    encoder_type = params['encoder']
    conv_type = params['conv']
    dd_decoder_type = params['decoder']
    h_sizes = params['hsize'] # only hidden and output_layer
    # nndecoder_h_size = params['nn_hsize']
    lr = params['lr']
    epochs = params['e']
    dr = params['dr']
    batch_size = params['batch']
    bias = params['bias']
    patience = params['patience']


    neg_fact = config_map['split']['neg_frac']
    number_of_folds = config_map['split']['folds']
    n_genes = len(gene_2_idx.keys())
    n_drugs = len( drug_target_df['pubchem_cid'].unique())
    total_cell_lines = len(cell_2_idx.values())
    drugs = drug_target_df['pubchem_cid'].unique()
    drug_2_idx = {node: i for i, node in enumerate(drugs)}
    idx_2_drug_node =  {i: node for i, node in enumerate(drugs)}

    #map drug, gene and cell line to index
    drug_target_df['gene_idx'] = drug_target_df['uniprot_id'].astype(str).apply(lambda x: gene_2_idx[x])
    drug_target_df['drug_idx'] = drug_target_df['pubchem_cid'].astype(str).apply(lambda x: drug_2_idx[x])

    synergy_df['Drug1_idx'] = synergy_df['Drug1_pubchem_cid'].astype(str).apply(lambda x: drug_2_idx[x])
    synergy_df['Drug2_idx'] = synergy_df['Drug2_pubchem_cid'].astype(str).apply(lambda x: drug_2_idx[x])
    synergy_df['Cell_line_idx'] = synergy_df['Cell_line'].astype(str).apply(lambda x: cell_2_idx[x])

    non_synergy_df['Drug1_idx'] = non_synergy_df['Drug1_pubchem_cid'].astype(str).apply(lambda x: drug_2_idx[x])
    non_synergy_df['Drug2_idx'] = non_synergy_df['Drug2_pubchem_cid'].astype(str).apply(lambda x: drug_2_idx[x])
    non_synergy_df['Cell_line_idx'] = non_synergy_df['Cell_line'].astype(str).apply(lambda x: cell_2_idx[x])


    # create and save all networks (in adjacency matrix format in a dict named adj_mats_init )
    adj_mats_init = {}
    adj_mats_init['gene_gene'] = create_gene_gene_network(ppi_sparse_matrix)
    adj_mats_init['drug_target'] , adj_mats_init['target_drug'] = create_drug_target_network(drug_target_df, gene_2_idx, drug_2_idx)
    adj_mats_init['drug_drug'] = create_drug_drug_network(synergy_df, n_drugs, total_cell_lines)


    ########    CROSS VALIDATION PREPARATION  ########

    folds_pos_drug_drug_edges, folds_neg_drug_drug_edges, \
    folds_pos_non_drug_drug_edges, folds_neg_non_drug_drug_edges =\
                            cross_val.prepare_cross_val_folds(folds_pos_drug_drug_edges, \
                                                    folds_neg_drug_drug_edges, \
                                                    synergy_df, non_synergy_df, adj_mats_init, neg_sampling_type, neg_fact, number_of_folds,
                                                    cross_val_dir)

    ######################## NODE FEATURE MATRIX CREATION ###########################################
    node_feat_dict = {}
    node_feat_dict['gene'] = torch.tensor(np.identity(n_genes)).to(dev) #featureless gene
    use_drug_feat_option = params['drugfeat']
    node_feat_dict['drug'] = prepare_drug_feat(drug_maccs_keys_feature_df, drug_2_idx, n_drugs, use_drug_feat_option).to(dev)




    ##########write training code here##################
    pos_df = pd.DataFrame()
    neg_df = pd.DataFrame()

    no_improvement_val_loss = 0
    model_no = 0
    machine_name = config_map['machine_name']
    for fold_no in range(number_of_folds):
        best_model_no = 0
        project_name = alg + '_' + machine_name
        with wandb.init(project=project_name, config=config_map):
            config_map = wandb.config
            ###################################### Prepare DATA ########################################

            train_pos_edges_dict, train_neg_edges_dict, val_pos_edges_dict, val_neg_edges_dict,val_es_pos_edges_dict,\
            val_es_neg_edges_dict, test_pos_edges_dict, test_neg_edges_dict =\
            prepare_train_edges(folds_pos_drug_drug_edges,
                                folds_neg_drug_drug_edges, folds_pos_non_drug_drug_edges,
                                folds_neg_non_drug_drug_edges, fold_no, total_cell_lines)

            # validation
            val_pos_edges_split_dict = {edge_type: [] for edge_type in edge_types}
            val_neg_edges_split_dict = {edge_type: [] for edge_type in edge_types}

            for edge_type in val_pos_edges_dict:
                for i in range(len(val_pos_edges_dict[edge_type])):
                    val_pos_edges_split_dict[edge_type].append(torch.split(val_pos_edges_dict[edge_type][i],
                                    batch_size, dim=1))
                    val_neg_edges_split_dict[edge_type].append(torch.split(val_neg_edges_dict[edge_type][i],
                                    batch_size * neg_fact, dim=1))


            #validation set for early stopping
            val_es_pos_edges_split_dict = {edge_type: [] for edge_type in edge_types}
            val_es_neg_edges_split_dict = {edge_type: [] for edge_type in edge_types}

            for edge_type in val_es_pos_edges_dict:
                for i in range(len(val_es_pos_edges_dict[edge_type])):
                    val_es_pos_edges_split_dict[edge_type].append(torch.split(val_es_pos_edges_dict[edge_type][i],
                                                                           batch_size, dim=1))
                    val_es_neg_edges_split_dict[edge_type].append(torch.split(val_es_neg_edges_dict[edge_type][i],
                                                                           batch_size * neg_fact, dim=1))

            #test data split only for drug_drug edges now
            test_pos_edges_split_dict = {edge_type: [] for edge_type in edge_types}
            test_neg_edges_split_dict = {edge_type: [] for edge_type in edge_types}

            for edge_type in test_pos_edges_dict:
                for i in range(len(test_pos_edges_dict[edge_type])):
                    test_pos_edges_split_dict[edge_type].append(torch.split(test_pos_edges_dict[edge_type][i],
                                    batch_size, dim=1))
                    test_neg_edges_split_dict[edge_type].append(torch.split(test_neg_edges_dict[edge_type][i],
                                    batch_size * neg_fact, dim=1))

            # t5 = time.time()
            # print('time for training matrix preparation: ', t5 - t4)

            # print('before split 1', train_pos_edges_dict['gene_gene'][0].size())
            edge_subtypes = {}
            for edge_type in edge_types:
                edge_subtypes[edge_type] = len(train_pos_edges_dict[edge_type])



            encoder = Encoder(h_sizes, bias , dr, encoder_type, conv_type, node_feat_dict, train_pos_edges_dict,
                              edge_types,edge_subtypes, n_drugs, n_genes)

            decoder_names = {'gene_gene': 'bilinear', 'target_drug': 'bilinear',
                             'drug_target': 'bilinear', 'drug_drug':dd_decoder_type}

            decoders = nn.ModuleDict()
            for edge_type in edge_types:
                n_sub_types = edge_subtypes[edge_type]

                if decoder_names[edge_type]=='bilinear':
                    decoders[edge_type] = BilinearDecoder(edge_type, n_sub_types, h_sizes[-1])

                elif decoder_names[edge_type]=='dedicom':
                    decoders[edge_type] = DedicomDecoder(edge_type, n_sub_types, h_sizes[-1])

                elif decoder_names[edge_type] == 'nn':
                    #the size of the input is concatenation of (drug1, drug2 , cell_line_spec_gene_expression)
                    #here, h_sizes[-1] is the output dimension of embedding layer i.e. final dim of embedded drugs
                    # and len(gene_expression_feature_df.columns) is the #of gene expression features
                    # nndecoder_hidden_layer_setup = synverse_params['nndecoder_hsize']
                    decoders[edge_type] = NNDecoder(edge_type, h_sizes[-1]*2 + len(gene_expression_feature_df.columns),
                                                    params['nn_hsize'] , gene_expression_feature_df)
                elif decoder_names[edge_type] == 'nn_nogenex':
                    decoders[edge_type] = NNDecoder_nogenex(edge_type, h_sizes[-1]*2, params['nn_hsize'])

            model = SynverseModel(encoder = encoder, decoders=decoders).to(dev)
            print('model_no: ',model_no)
            print('fold_no: ',fold_no)
            model_no +=1

            # model_param = model.state_dict()
            # model_param = model.parameters()
            # print(model_param)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            print('learning rate: ', lr, 'h_sizes: ', h_sizes, 'optimizer: ', optimizer)

            minibatch_handlder = MinibatchHandler(train_pos_edges_dict, batch_size, total_cell_lines)

            wandb.watch(model, log='all', log_freq=1)

            wandb_step = 0

            min_loss_dd_val_es = 1000
            min_loss_dd_val = 1000

            for epoch in range(1, epochs+1):
                t10 = time.time()
                #shuffle each training tensor  at the beginning of each epoch


                train_pos_edges_dict = minibatch_handlder.shuffle_train_edges(train_pos_edges_dict)
                train_neg_edges_dict = minibatch_handlder.shuffle_train_edges(train_neg_edges_dict)


                #split the train edges in chunks with size=batch_size
                #dict of list of split tensors/torches
                train_pos_edges_split_dict = {edge_type:[] for edge_type in edge_types}
                train_neg_edges_split_dict = {edge_type:[] for edge_type in edge_types}

                for edge_type in train_pos_edges_dict:
                    for i in range(len(train_pos_edges_dict[edge_type])):
                        train_pos_edges_split_dict[edge_type].append(torch.split(train_pos_edges_dict[edge_type][i],
                                                                               batch_size, dim=1))
                        train_neg_edges_split_dict[edge_type].append(torch.split(train_neg_edges_dict[edge_type][i],
                                                                               batch_size * neg_fact, dim=1))


                while True:
                    if use_drug_based_batch_end:
                        if minibatch_handlder.is_batch_finished_new():
                            break
                    else:
                        if minibatch_handlder.is_batch_finished():
                            break

                    # t13 = time.time()
                    e, edge_sub_type_idx, batch_num = minibatch_handlder.next_minibatch()

                    batch_pos_train_edges = train_pos_edges_split_dict[e][edge_sub_type_idx][batch_num].to(dev)

                    batch_neg_train_edges = train_neg_edges_split_dict[e][edge_sub_type_idx][batch_num].to(dev)
                    training_batch_loss  = train(model, optimizer, batch_pos_train_edges, batch_neg_train_edges,
                                                e, edge_sub_type_idx)

                    wandb_step+=1
                    if e == 'drug_drug':
                        cell_line = idx_2_cell[edge_sub_type_idx]
                        train_log(training_batch_loss, wandb_step, e, cell_line)
                    if e == 'gene_gene':
                        # wandb_step += 1
                        train_log(training_batch_loss, wandb_step, e, 'gene_gene')

                    wandb.log({'Epoch': epoch}, step=wandb_step)

                # plot_loss((avg_loss, edge_type, train_or_val, wandb_step)
                ##train and val_es loss plot after whole epoch
                if epoch % 2 == 0:
                    loss_dd_train = compute_loss(model, 'drug_drug', train_pos_edges_split_dict, train_neg_edges_split_dict,
                          edge_subtypes)
                    plot_loss(loss_dd_train, 'drug_drug', 'train', wandb_step)
                    # loss_gg_train = compute_loss(model, 'gene_gene', train_pos_edges_split_dict, train_neg_edges_split_dict,
                    #                                           edge_subtypes)
                    # plot_loss(loss_gg_train, 'gene_gene', 'train', wandb_step)
                    #
                    # loss_dg_train = compute_loss(model, 'drug_target', train_pos_edges_split_dict,train_neg_edges_split_dict,
                    #                                           edge_subtypes)
                    # plot_loss(loss_dg_train, 'drug_target', 'train', wandb_step)
                    #loss_gd_train = compute_loss(model, 'target_drug', train_pos_edges_split_dict, train_neg_edges_split_dict,
                    #                                           edge_subtypes)
                    #plot_loss(loss_gd_train, 'target_drug', 'train', wandb_step)

                    # print('find memory consumption in loss_dd_val: before')
                    # gpu_memory_usage()
                    #
                    #
                    loss_dd_val_es = compute_loss(model, 'drug_drug', val_es_pos_edges_split_dict, val_es_neg_edges_split_dict,
                                                           edge_subtypes)
                    plot_loss(loss_dd_val_es, 'drug_drug', 'val_es', wandb_step)

                    # # print('after: ')
                    # # gpu_memory_usage()
                    #
                    # loss_dd_val = compute_loss(model, 'drug_drug', val_pos_edges_split_dict,
                    #                                        val_neg_edges_split_dict,
                    #                                        edge_subtypes)
                    # loss_gg_val = compute_loss(model, 'gene_gene', val_pos_edges_split_dict, val_neg_edges_split_dict,
                    #                                         edge_subtypes)

                    # loss_dg_val = compute_loss(model, 'drug_target', val_pos_edges_split_dict,val_neg_edges_split_dict,
                    #                                         edge_subtypes)
                    #
                    # loss_gd_val = compute_loss(model, 'target_drug', val_pos_edges_split_dict, val_neg_edges_split_dict,
                    #                                         edge_subtypes)

                    # loss_dd_val_es = compute_loss(model, 'drug_drug', val_es_pos_edges_split_dict, val_es_neg_edges_split_dict,
                    #                                edge_subtypes)

                    if loss_dd_val_es < min_loss_dd_val_es:

                        best_model_path = save_best_model(model, params, out_dir)

                        required_epoch = epoch
                        min_loss_dd_val_es = loss_dd_val_es
                        # min_loss_dd_val = loss_dd_val
                        no_improvement_val_loss = 0
                        print('saving new model', best_model_no)
                        best_model_no +=1

                    if loss_dd_val_es >= min_loss_dd_val_es:
                        no_improvement_val_loss += 1
                        if no_improvement_val_loss > patience:
                            #save the model with loss on 'val' dataset. depending on this loss value I will decide on the
                            #best hyperparamater later.
                            del model
                            gc.collect()
                            break

                print('epoch: ', epoch, ' epoch time: ', time.time() - t10)

            #save test result from the best model here
            best_model = SynverseModel(encoder = encoder, decoders=decoders)
            best_model.load_state_dict(torch.load(best_model_path))
            best_model.to(dev)
            best_model_dd_val_loss = compute_loss(best_model, 'drug_drug', val_pos_edges_split_dict, \
                                                  val_neg_edges_split_dict, edge_subtypes)
            model_utils.save_model_info_with_loss(best_model_dd_val_loss, params, out_dir, fold_no)

            test_edge_type = 'drug_drug'
            # total_test_loss = 0
            for edge_sub_type in range(edge_subtypes[test_edge_type]):
                cell_line = idx_2_cell[edge_sub_type]
                for split_idx in range(len(test_pos_edges_split_dict[test_edge_type][edge_sub_type])):
                    test_pos_edges = test_pos_edges_split_dict[test_edge_type][edge_sub_type][split_idx].to(dev)
                    test_neg_edges = test_neg_edges_split_dict[test_edge_type][edge_sub_type][split_idx].to(dev)

                    if test_pos_edges.size()[1] != 0:
                        batch_wise_pos_pred, batch_wise_neg_pred, test_loss = val(best_model, test_pos_edges, test_neg_edges,\
                                                                                 test_edge_type, edge_sub_type)
                        # total_test_loss += test_loss

                        batch_wise_pos_df, batch_wise_neg_df = prepare_pred_score_for_saving\
                            (batch_wise_pos_pred, batch_wise_neg_pred, cell_line, idx_2_drug_node)
                        pos_df = pd.concat([pos_df, batch_wise_pos_df], axis=0)
                        neg_df = pd.concat([neg_df, batch_wise_neg_df], axis=0)

            del best_model
            gc.collect()
            if not do_save_best_model:
                os.remove(best_model_path)


        model_utils.save_drug_drug_link_probability(pos_df, neg_df, params, out_dir)


    #write code for
    gc.collect()
