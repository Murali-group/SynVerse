import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        output_size = 1
        in_dropout_rate = config['in_dropout_rate']
        hid_dropout_rate = config['hid_dropout_rate']

        # initiate and train model
        hidden_layers = []
        for i in range(config['num_hid_layers']):
            hidden_layers.append(config[f'hid_{i}'])

        self.layers = nn.ModuleList()
        for i in range(len(hidden_layers)):
            self.layers.append(nn.Linear(input_size if i == 0 else hidden_layers[i - 1], hidden_layers[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(in_dropout_rate if i == 0 else hid_dropout_rate))

        self.layers.append(nn.Linear(hidden_layers[-1], output_size))

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x