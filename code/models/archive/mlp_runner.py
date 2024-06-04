from utils import *
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from models.mlp_wrapper import *
from models.model_utils import *
import os

def MLP_runner(train_val_df, train_idx, val_idx, test_df, dfeat_dict, cfeat_dict, drug_2_idx, cell_line_2_idx,out_file_prefix, is_wandb, device):
    node_X = concatenate_features(dfeat_dict, 'pid', drug_2_idx)
    edge_type_X = concatenate_features(cfeat_dict, 'cell_line_name', cell_line_2_idx)
    init_size= len(train_val_df)
    X_train_val = torch.from_numpy(prepare_feat_for_MLP(train_val_df[['source', 'target', 'edge_type']], node_X, edge_type_X))
    y_train_val = torch.tensor(list(train_val_df['S_mean_mean'].values)*2) #repeat labels

    #Test data
    X_test_val = torch.from_numpy(prepare_feat_for_MLP(test_df[['source', 'target', 'edge_type']], node_X, edge_type_X))
    y_test_val = torch.tensor(list(test_df['S_mean_mean'].values) * 2)  # repeat labels

    #****************** model running parameters
    batch_size= 4096
    n_epochs = 1500
    check_freq = 5  #after how many epocks I should check if the validation loss is decreasing or not.
    tolerance = 10 #if for tolerance number of epochs the validation loss does not decrease, then stop training and return the then best_model.
    #****************** model architecture parameters
    lr=0.0001
    # hidden_layers = [8192, 4096]
    hidden_layers = [2048, 512]
    in_dropout_rate=0.2
    hid_dropout_rate=0.5

    val_losses = {}
    test_losses={}
    dataset= TensorDataset(X_train_val, y_train_val)
    test_dataset = TensorDataset(X_test_val, y_test_val)
    n_folds = len(val_idx.keys())

    out_file=out_file_prefix+'.txt'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    open(out_file,'w')
    for fold in range(n_folds):
        train_ids = train_idx[fold] + [a+init_size for a in train_idx[fold]]
        val_ids = val_idx[fold] + [a+init_size for a in val_idx[fold]]

        train_subsampler = Subset(dataset, train_ids)
        val_subsampler = Subset(dataset, val_ids)

        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = MLP_wrapper(input_size=X_train_val.shape[1], hidden_layers=hidden_layers,
                            output_size=1, in_dropout_rate=in_dropout_rate, hid_dropout_rate=hid_dropout_rate).to(device)

        ## Wrap the model for parallel processing if multiple gpus are available
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_model_state, val_losses[fold] = train_model(model, optimizer, criterion, train_loader, val_loader, n_epochs,
                                              check_freq, tolerance,is_wandb, device)
        print(f'\nVAL MSE loss in fold {fold}: {val_losses[fold]}')

        # based on the loss for different parameter setup we have to identify the best model and get evaluate that
        # on test data
        model.load_state_dict(best_model_state)
        test_losses[fold] = eval_model(model, test_loader, criterion, device)
        print(f'TEST MSE loss in fold {fold}: {test_losses[fold]}\n')

        with open (out_file,'a') as file:
            file.write(f'val_loss: {val_losses[fold]}\ntest_loss: {test_losses[fold]}\n\n')
        file.close()

    del(X_train_val, y_train_val, X_test_val, y_test_val)

    return val_losses, test_losses

