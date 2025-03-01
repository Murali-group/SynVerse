import torch
import torch.nn as nn
import math
torch.set_default_dtype(torch.float32)

def get_sinusoidal_positional_encoding(seq_len, d_model, device):
    """Generate sinusoidal positional encodings."""
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0)  # Add a batch dimension

def pad_or_truncate(seq, max_len):

    if len(seq) < max_len:
        return seq + [0] * (max_len - len(seq))
    else:
        return seq[:max_len]

#     # smiles_df['tokenized'] = smiles_df['tokenized'].apply(lambda x: pad_or_truncate(x, max_len))

class Transformer_Encoder(nn.Module):
    def __init__(self, vocab_size, config, device):
        super().__init__()
        self.device = device

        self.max_seq_length = config['max_seq_length']
        self.d_model = config['transformer_embedding_dim']
        self.n_head = config['transformer_n_head']
        self.transformer_n_layers = config['transformer_num_layers']
        self.dim_feedforward = config['transformer_ff_num_layers']
        self.positional_encoding_type = config['positional_encoding_type']

        self.batch_norm = config['transformer_batch_norm']

        self.embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=0)

        if self.positional_encoding_type == 'learnable':
            self.pos_encoder = nn.Embedding(self.max_seq_length, self.d_model)
        elif self.positional_encoding_type == 'fixed':
            self.positional_encoding = get_sinusoidal_positional_encoding(self.max_seq_length, self.d_model, self.device)

        encoder_layers = nn.TransformerEncoderLayer(self.d_model, self.n_head, self.dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.transformer_n_layers)

        self.out_dim = self.d_model

    def forward(self, src):
        #padding or truncating
        src = [pad_or_truncate(x, self.max_seq_length) for x in src]
        # Convert each element to a tensor and move it to the device
        src = torch.stack([torch.tensor(element, dtype=torch.long) for element in src]).to(self.device)

        # Get token embeddings
        src_embed = self.embedding(src)

        # Add positional encodings
        if self.positional_encoding_type == 'learnable':
            positions = torch.arange(0, src_embed.size(1), device=src_embed.device).unsqueeze(0).repeat(
                src_embed.size(0), 1)
            src_embed += self.pos_encoder(positions)
        elif self.positional_encoding_type == 'fixed':
            src_embed += self.positional_encoding[:, :src_embed.size(1), :].to(self.device)

        # Pass through the transformer encoder
        output = self.transformer_encoder(src_embed.transpose(0, 1))

        # Use the output corresponding to the first token ([CLS] token)
        output = output[0, :, :]

        return output.squeeze()
