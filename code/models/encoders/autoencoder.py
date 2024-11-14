import torch.nn as nn
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        '''
        A 2 layered symmetric MLP
        '''
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
        )
        self.reconstruct_loss = nn.L1Loss(reduction='mean')

    def compute_loss(self,true, pred):
        return self.reconstruct_loss(pred, true)

    def forward(self, x, decoder=True):
        embedding = self.encoder(x)
        if decoder:
            reconstructed = self.decoder(embedding)
            return embedding, reconstructed
        return embedding


