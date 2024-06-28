import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_sinusoidal_positional_encoding(seq_len, d_model, device):
    """Generate sinusoidal positional encodings."""
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0)  # Add a batch dimension

class Transformer_Encoder(nn.Module):
    def __init__(self, vocab_size, max_seq_length, device, config):
        super().__init__()
        self.d_model = config['transformer_embed_dimension']
        self.n_head = config['transformer_n_head']
        self.transformer_n_layers = config['transformer_num_layers']
        self.dim_feedforward = config['ff_num_layers']
        self.positional_encoding_type = config['positional_encoding_type']
        self.device = device

        self.batch_norm = config['batch_norm']

        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)

        if self.positional_encoding_type == 'learnable':
            self.pos_encoder = nn.Embedding(self.max_seq_length, self.d_model)
        elif self.positional_encoding_type == 'fixed':
            self.positional_encoding = get_sinusoidal_positional_encoding(self.max_seq_length, self.d_model, self.device)

        encoder_layers = nn.TransformerEncoderLayer(self.d_model, self.n_head, self.dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.transformer_n_layers)
        self.fc = nn.Linear(self.d_model, 1)

    def forward(self, src):
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
        print(f'output for {src} after {self.transformer_n_layers} layers of transformer: {output}')

        # Use the output corresponding to the first token ([CLS] token)
        output = self.fc(output[0, :, :])
        return output.squeeze()
