'''
Define the mlp model here.
'''
import copy
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

from models.encoders.drug.gcn_encoder import GCN_Encoder
from models.encoders.drug.transformer_encoder import Transformer_Encoder

from models.decoders.mlp import MLP

class Encoder_MLP_wrapper(nn.Module):
    def __init__(self, drug_encoder_list, cell_encoder_list, dfeat_dim_dict, cfeat_dim_dict,
                 drug_feat_encoder_mapping, cell_feat_encoder_mapping, config, device):
        super().__init__()
        self.chosen_config = config

        self.dfeat_dim_dict = dfeat_dim_dict
        self.cfeat_dim_dict = cfeat_dim_dict

        self.dfeat_out_dim = copy.deepcopy(dfeat_dim_dict)
        self.cfeat_out_dim = copy.deepcopy(cfeat_dim_dict)

        self.drug_feat_encoder_mapping = drug_feat_encoder_mapping
        self.cell_feat_encoder_mapping =cell_feat_encoder_mapping
        self.drug_encoder_list = drug_encoder_list if drug_encoder_list is not None else []
        self.cell_encoder_list = cell_encoder_list if cell_encoder_list is not None else []
        self.device = device

        #drug encoder
        for feat_name in drug_feat_encoder_mapping:
            encoder_name = drug_feat_encoder_mapping[feat_name]
            for drug_encoder in self.drug_encoder_list:
                if (drug_encoder['name'] == encoder_name):

                    if encoder_name=='GCN':
                        self.gcn_encoder = GCN_Encoder(self.dfeat_dim_dict[feat_name], config)
                        #update the drug feat dim with the dimension of generated embedding
                        self.dfeat_out_dim[feat_name] = self.gcn_encoder.out_dim

                    if encoder_name == 'Transformer':
                        # print(self.chosen_config)
                        self.transformer_encoder = Transformer_Encoder(self.dfeat_dim_dict[feat_name], config, self.device)
                        # update the drug feat dim with the dimension of generated embedding
                        self.dfeat_out_dim[feat_name] = self.transformer_encoder.out_dim

                    # if encoder_name == 'SPMM':
                    #     # print(self.chosen_config)
                    #     self.SPMM_encoder = SPMM_Encoder(drug_encoder['params']['vocab'], drug_encoder['params']['checkpoint'],config, self.device)
                    #     # update the drug feat dim with the dimension of generated embedding
                    #     self.dfeat_out_dim[feat_name] = self.SPMM_encoder.out_dim

        #TODO: other drug encoders.

        #TODO: cell line encoder.


        #MLP synergy predictor
        # input_size = 8451 #TODO figure out input_size here
        drug_dim = 0
        cell_dim = 0
        for feat_name in self.dfeat_out_dim:
            drug_dim += self.dfeat_out_dim[feat_name]
        for feat_name in self.cfeat_out_dim:
            cell_dim+= self.cfeat_out_dim[feat_name]

        input_size = drug_dim*2+cell_dim

        #The synergy predictor MLP
        self.mlp = MLP(input_size, config)

    def drug_encoder_wrap(self, drug_feat, batch_drugs,  device):
        # For each ( feature_name: encoder, e.g., smiles:GCN ) from self.drug_encoder_dict
        # pass feature matrix(smiles) to GCN and get embedding of drugs.
        #return a dict with key='feature_name-encoder', e.g., 'smiles_GCN' and value=embedding.
        drug_represenatation = []
        embedded_feat=[]

        for feat_name in self.drug_feat_encoder_mapping:
            encoder_name = self.drug_feat_encoder_mapping[feat_name]
            for drug_encoder in self.drug_encoder_list:
                if (drug_encoder['name'] == encoder_name):
                    if encoder_name=='GCN':
                        # create a list of drug_graphs where in index i, the molecular graph of drug i is present.
                        #TODO check if filtering out the batch_drugs is done properly.
                        # data_list = [drug_feat[feat_name][x] for x in range(len(drug_feat[feat_name].keys()))][batch_drugs]
                        data_list = [drug_feat[feat_name][x] for x in batch_drugs]
                        drug_represenatation.append(self.gcn_encoder(data_list, device))
                        embedded_feat.append(feat_name)

                    if encoder_name=='Transformer':
                        source = drug_feat[feat_name][batch_drugs,0]
                        drug_represenatation.append(self.transformer_encoder(source))
                        embedded_feat.append(feat_name)

                    # if encoder_name=='SPMM':
                    #     drug_smiles = drug_feat[feat_name][batch_drugs,0]
                    #     drug_represenatation.append(self.SPMM_encoder(drug_smiles))
                    #     embedded_feat.append(feat_name)

                    #TODO add more encoder here

        #now concatenate any raw drug features present in drug_feat
        for feat_name in drug_feat:
            if feat_name not in embedded_feat: #features for which no encoder is given
                drug_represenatation.append(torch.from_numpy(drug_feat[feat_name][batch_drugs]).to(device))

        #get the final features of drugs by concatenating both embedding and raw features
        drug_final_embeds = torch.cat(drug_represenatation, dim=1).to(device)
        return drug_final_embeds

    def cell_line_encoder_wrap(self, cell_line_feat, batch_cell_lines, device):
        # For each ( feature_name: encoder, e.g., genex:autoencoder ) from self.cell_line_encoder_dict
        # pass feature matrix(genex) to autoencoder and get embedding of drugs.
        # return a dict with key='feature_name-encoder', e.g., 'genex_autoencoder' and value=embedding.

        # now concatenate any raw cell features present in cell_line_feat
        cell_embed_raw_feats = []
        embedded_feat=[]
        for feat_name in cell_line_feat:
            # if (self.cell_encoder_dict is not None):
            if feat_name not in embedded_feat:  # features for which no encoder is given
                cell_embed_raw_feats.append(torch.from_numpy(cell_line_feat[feat_name][batch_cell_lines]).to(device))
        # concat raw feats of drugs
        cell_final_embeds = torch.cat(cell_embed_raw_feats, dim=1).to(device)
        # get the final features of drugs by concatenating both embedding and raw features
        return cell_final_embeds

    def concat_feat(self, batch_triplets, batch_drugs, batch_cell_lines, drug_X, cell_X):
        '''

        '''
        # give drugs and cell line in batch_drugs and batch_cell_lines a local index and use that index in batch_triplets later.
        batchwise_drug_idx = {drug: i for (i, drug) in enumerate(batch_drugs)}
        batchwise_cell_idx = {cell: i for (i, cell) in enumerate(batch_cell_lines)}

        drug1s = batch_triplets[:, 0].flatten().numpy()
        drug2s = batch_triplets[:, 1].flatten().numpy()
        cell_lines = batch_triplets[:, 2].flatten().numpy()

        batch_indexed_drug1s = torch.tensor([batchwise_drug_idx[d] for d in drug1s]).to(self.device)
        batch_indexed_drug2s = torch.tensor([batchwise_drug_idx[d] for d in drug2s]).to(self.device)
        batch_indexed_cell_lines = torch.tensor([batchwise_cell_idx[c] for c in cell_lines]).to(self.device)

        # Use indexing to fetch all necessary features directly
        source_features = drug_X[batch_indexed_drug1s]
        target_features = drug_X[batch_indexed_drug2s]
        edge_features = cell_X[batch_indexed_cell_lines]
        # Concatenate the features along the second axis (column-wise concatenation)
        # concatenation: source-target-edge_type
        # concatenated_features = np.concatenate([source_features, target_features, edge_features], axis=1)
        mlp_ready_feats = torch.cat((source_features, target_features, edge_features), dim=1)
        return mlp_ready_feats

    def forward(self, batch_triplets, drug_feat, cell_line_feat, device):

        #batch_drugs: find out the drugs in the current batch
        #batch_cell_lines: find out the drugs in the cu
        batch_drugs = list(set(batch_triplets[:, 0].flatten().numpy()).union(set(batch_triplets[:, 1].flatten().numpy())))
        batch_cell_lines = list(set(batch_triplets[:, 2].flatten().numpy()))

        drug_embeds = self.drug_encoder_wrap(drug_feat, batch_drugs, device)
        cell_embeds = self.cell_line_encoder_wrap(cell_line_feat, batch_cell_lines, device)

        x = self.concat_feat(batch_triplets, batch_drugs,batch_cell_lines, drug_embeds, cell_embeds)
        x = x.to(device)
        x = self.mlp(x)
        return x

    def number_of_parameters(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))

