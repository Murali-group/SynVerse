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
    """Pad sequences with zeros or truncate to max_len."""
    if len(seq) < max_len:
        return seq + [0] * (max_len - len(seq))
    else:
        return seq[:max_len]

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
        self.pooling_type = config.get('pooling_type', 'CLS')
        self.embedding_dropout = config.get('embedding_dropout', 0)
        self.dropout = config.get('dropout', 0.1)
        self.batch_norm = config.get('transformer_batch_norm')

        self.embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=0)
        self.embed_dropout = nn.Dropout(self.embedding_dropout)

        if self.positional_encoding_type == 'learnable':
            self.pos_encoder = nn.Embedding(self.max_seq_length, self.d_model)
        elif self.positional_encoding_type == 'fixed':
            self.register_buffer('positional_encoding',
                                 get_sinusoidal_positional_encoding(self.max_seq_length, self.d_model, device))

        # Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='gelu', # Using GELU instead of ReLU (similar to BERT)
            batch_first=True, # Input and output tensors will have shape (batch_size, seq_len, d_model)
            norm_first=True # Apply layer norm before attention and FF (similar to BERT)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.transformer_n_layers)

        self.out_dim = self.d_model

        # Batch normalization for the output embeddings
        if self.batch_norm:
            self.output_norm = nn.BatchNorm1d(self.d_model)

        # For attention pooling if used
        if self.pooling_type == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(self.d_model, 1),
                nn.Softmax(dim=1)
            )

    def forward(self, src):
        # Pad/truncate sequences
        src = [pad_or_truncate(x, self.max_seq_length) for x in src]
        src = (torch.stack([torch.tensor(element, dtype=torch.long) for element in src])
               .to(self.device)) # [batch_size, seq_len]

        # Create padding mask (True for padding positions, False for token positions)
        padding_mask = (src == 0)

        # Get token embeddings
        src_embed = self.embedding(src)
        src_embed = self.embed_dropout(src_embed)

        # Add positional encodings
        if self.positional_encoding_type == 'learnable':
            positions = (torch.arange(0, src_embed.size(1), device=self.device)
                         .unsqueeze(0).repeat(src_embed.size(0), 1))
            src_embed += self.pos_encoder(positions)
        elif self.positional_encoding_type == 'fixed':
            src_embed += self.positional_encoding[:, :src_embed.size(1), :]

        # Pass through the transformer encoder with padding mask
        # Padding positions should be ignored during self-attention
        output = self.transformer_encoder(src_embed, src_key_padding_mask=padding_mask) # [batch_size, seq_len, d_model]

        # Pool the embeddings based on pooling type
        if self.pooling_type == 'CLS':
            # Use the [CLS] token (first position) embedding
            pooled_embedding = output[:, 0, :] # [batch_size, d_model]

        elif self.pooling_type == 'mean':
            # a boolean mask for valid (non-padding) tokens
            mask = ~padding_mask #  mask for valid tokens
            # Sum and divide by number of valid tokens (excluding padding)
            pooled_embedding = torch.sum(output * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1, keepdim=True) # [batch_size, d_model]

        elif self.pooling_type == 'attention':
            # Attention-weighted pooling
            attention_weights = self.attention_pool(output).transpose(1, 2) # [batch_size, 1, seq_len]
            # Zero out padding tokens in attention
            attention_weights = attention_weights.masked_fill(padding_mask.unsqueeze(1), float('-inf'))
            # Apply softmax after masking
            attention_weights = torch.softmax(attention_weights, dim=-1) # [batch_size, 1, seq_len]
            # Perform the weighted pooling using the attention weights
            pooled_embedding = torch.bmm(attention_weights, output) # [batch_size, 1, d_model]
            pooled_embedding = pooled_embedding.squeeze(1)  # [batch_size, d_model]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # Apply batch normalization if specified
        return self.output_norm(pooled_embedding) if self.batch_norm else pooled_embedding
