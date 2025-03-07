import os.path

import torch
import torch.nn as nn
import dgl

# the pretrained model and related modules
from models.pretrained.kpgt.src.model.light import LiGhTPredictor as LiGhT
from models.pretrained.kpgt.src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES

# def set_seed(seed):
#     """Set seed for reproducibility."""
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def init_params(module):
#     """Initialize parameters for Linear and Embedding layers."""
#     if isinstance(module, nn.Linear):
#         module.weight.data.normal_(0.0, 0.02)
#         if module.bias is not None:
#             module.bias.data.zero_()
#     if isinstance(module, nn.Embedding):
#         module.weight.data.normal_(0.0, 0.02)

# def get_predictor(d_input_feats, n_tasks, n_layers, predictor_drop, device, d_hidden_feats=None):
#     if n_layers == 1:
#         predictor = nn.Linear(d_input_feats, n_tasks)
#     else:
#         predictor = nn.ModuleList()
#         predictor.append(nn.Linear(d_input_feats, d_hidden_feats))
#         predictor.append(nn.Dropout(predictor_drop))
#         predictor.append(nn.GELU())
#         for _ in range(n_layers - 2):
#             predictor.append(nn.Linear(d_hidden_feats, d_hidden_feats))
#             predictor.append(nn.Dropout(predictor_drop))
#             predictor.append(nn.GELU())
#         predictor.append(nn.Linear(d_hidden_feats, n_tasks))
#         predictor = nn.Sequential(*predictor)
#     predictor.apply(lambda module: init_params(module))
#     return predictor.to(device)

"""
This model uses a pretrained LiGhT model as an encoder for drug embeddings,
"""
class KPGT_Encoder(nn.Module):
    def __init__(self, config, device):
        super(KPGT_Encoder, self).__init__()
        self.device = device

        # Initialize vocabulary for molecular features
        self.vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
        self.root_path = config['pre_train_root_path']
        self.dataset = config['dataset']
        self.pretrained_model_path = os.path.join(self.root_path, config['pretrained_model'])

        # Build the LiGhT encoder using the provided configuration
        self.encoder = LiGhT(
            d_node_feats=config['d_node_feats'],
            d_edge_feats=config['d_edge_feats'],
            d_g_feats=config['d_g_feats'],
            d_fp_feats=config['d_fp_feats'],
            d_md_feats=config['d_md_feats'],
            d_hpath_ratio=config['d_hpath_ratio'],
            n_mol_layers=config['n_mol_layers'],
            path_length=config['path_length'],
            n_heads=config['n_heads'],
            n_ffn_dense_layers=config['n_ffn_dense_layers'],
            input_drop=config.get('input_drop', 0),
            attn_drop=config.get('attn_drop', 0),
            feat_drop=config.get('feat_drop', 0),
            n_node_types=self.vocab.vocab_size
        ).to(device)

        # Load pretrained weights into the encoder.
        state_dict = torch.load(self.pretrained_model_path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.encoder.load_state_dict(state_dict)

        # Remove the original prediction heads, so that only the encoder is used.
        del self.encoder.md_predictor
        del self.encoder.fp_predictor
        del self.encoder.node_predictor

        # Attach a new head for the downstream task.
        # self.encoder.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 3, n_tasks=config['dim_out'],
        #                                 n_layers=config['n_predictor_layers'], predictor_drop=0, device=device,
        #                                 d_hidden_feats=config['d_predictor_hidden'])

        print("Total model parameters: {:.2f}M".format(sum(x.numel() for x in self.encoder.parameters()) / 1e6))

        self.out_dim = config['dim_out']

    def forward(self, data):
        # Pass input data through the encoder to get an internal embeddings.
        batched_graph = dgl.batch(data.graphs)  # batch the graphs for each drug separately

        embeddings = self.encoder(batched_graph.to(self.device), data.fps.to(self.device), data.mds.to(self.device))
        return embeddings
