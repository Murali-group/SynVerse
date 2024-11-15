from models.encoders.autoencoder import AutoEncoder
import torch
import os
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
import numpy as np

def train_autoencoder(feat, hidden_dims, epoch, model_file, device):
    '''
    Given a feature matrix feat, return the embedding matrix
    '''
    feat_tensor = torch.tensor(feat, dtype=torch.float32)
    dataset = TensorDataset(feat_tensor)

    model = AutoEncoder(feat_tensor.shape[1], hidden_dims).to(torch.float32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    trn_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    for i in range(1, epoch + 1):
        model.train()
        for x_batch, in trn_loader:
            x_batch = x_batch.to(device)
            model_outs = model(x_batch)
            trn_loss = model.compute_loss(x_batch, model_outs[1])
            optimizer.zero_grad()
            trn_loss.backward()
            optimizer.step()
            # print(f"Epoch {i:03d} | Train Loss: {trn_loss.item():.6f}")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    torch.save(model.state_dict(), model_file)


def train_cv(feat_tensor, input_dim, hidden_dims, epoch, patience=10, device='cpu'):
    dataset = TensorDataset(feat_tensor)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_losses = []
    fold_epochs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")

        # Split data into train and validation sets
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        # DataLoaders for training and validation sets
        trn_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

        # Initialize model
        model = AutoEncoder(input_dim, hidden_dims).to(torch.float32).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

        best_val_loss = float('inf')
        epoch_req = epoch
        epochs_no_improve = 0

        for i in range(1, epoch + 1):
            # Training step
            model.train()
            for x_batch, in trn_loader:
                x_batch = x_batch.to(device)
                model_outs = model(x_batch)
                trn_loss = model.compute_loss(x_batch, model_outs[1])
                optimizer.zero_grad()
                trn_loss.backward()
                optimizer.step()
            # print(f"Epoch {i:03d} | Train Loss: {trn_loss.item():.6f}")

            # Validation step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, in val_loader:
                    x_batch = x_batch.to(device)
                    model_outs = model(x_batch)
                    batch_loss = model.compute_loss(x_batch, model_outs[1])
                    val_loss += batch_loss.item()  # Moving loss to CPU and summing up

            avg_val_loss = val_loss / len(val_loader)
            # print(f"Epoch {i:03d} | Validation Loss: {avg_val_loss:.6f}")

            # Check if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                epoch_req=i
                # Save the best model
                # torch.save(model.state_dict(), f'{model_file}_fold{fold + 1}_best.pth')
            else:
                epochs_no_improve += 1

            # Early stopping check
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {i} | Best Validation Loss: {best_val_loss:.6f}")
                break

        # Record the best validation loss for this fold
        fold_losses.append(best_val_loss)
        fold_epochs.append(epoch_req)
        print(f"Fold {fold + 1} | Best Validation Loss: {best_val_loss:.6f}")

    # Average validation loss across all folds
    avg_fold_loss = np.mean(fold_losses)
    avg_req_epoch = int(np.mean(fold_epochs))

    print(f"Average Validation Loss across 5 folds: {avg_fold_loss:.6f}")
    return avg_fold_loss, avg_req_epoch


def tune_hyperparam(feat, hidden_dim_options, epoch, device, patience=20):
    '''
    Given a feature matrix feat, perform 5-fold cross-validation with early stopping for hyperparameter tuning.
    Returns the average validation loss across folds.
    '''

    feat_tensor = torch.tensor(feat, dtype=torch.float32)
    input_dim = feat_tensor.shape[1]

    best_val_loss = float('inf')
    best_hidden_dim = []
    best_epochs = 0
    for hidden_dims in hidden_dim_options:
        avg_val_loss, req_epoch = train_cv(feat_tensor, input_dim, hidden_dims, epoch, patience, device=device)

        if avg_val_loss<best_val_loss:
            best_val_loss = avg_val_loss
            best_hidden_dim = hidden_dims
            best_epochs = req_epoch

    return best_hidden_dim, best_epochs




def get_embedding(feat, hidden_dims, model_file, device):
    feat_tensor = torch.tensor(feat, dtype=torch.float32)
    dataset = TensorDataset(feat_tensor)
    test_loader = DataLoader(dataset, batch_size=len(feat), shuffle=False)


    model = AutoEncoder(feat_tensor.shape[1], hidden_dims).to(torch.float32).to(device)
    model_file = model_file


    # Evaluate and get reduced representations
    reduced_feat = []  # List to collect reduced dimensions
    with torch.no_grad():
        model.eval()
        model.load_state_dict(torch.load(model_file))
        for x_batch, in test_loader:
            x_batch = x_batch.to(device)
            embedding = model(x_batch, decoder=False)  # Get embedding without decoding
            reduced_feat.append(embedding.cpu().numpy())  # Append to list

    # Stack reduced_feat to get the final reduced array with the same row order
    reduced_feat = np.vstack(reduced_feat)
    return reduced_feat



def autoencoder_wrapper(feat_mtx_dict, feat_dim_dict, feat_compress_info, train_drug_idx, hidden_dim_options, epoch = 100, file_prefix=None, device='cpu', force_run=True):
    for feat_name in feat_compress_info:
        if feat_compress_info[feat_name]: #if compress feature = True
            model_file = f'{file_prefix}/{feat_name}.pth'
            feature_file = f'{file_prefix}/{feat_name}.npy'
            if (not os.path.exists(feature_file)) or (force_run):  # if model file doesn't exist, train and save the model
                feat_mtx_train = feat_mtx_dict[feat_name][train_drug_idx, :]  # keep the training data only
                #train autoencoder using only the drugs present in training split
                best_hidden_dims, req_epochs = tune_hyperparam(feat_mtx_train, hidden_dim_options, epoch, device)
                #train model with best hidden dim and epochs and save model in model_file
                train_autoencoder(feat_mtx_train, best_hidden_dims, req_epochs, model_file, device)

                #get embedding for all drugs/cell lines including those present in test data
                feat_mtx_dict[feat_name] = get_embedding(feat_mtx_dict[feat_name], best_hidden_dims, model_file, device)
                feat_dim_dict[feat_name] = feat_mtx_dict[feat_name].shape[1] #update feature dimension with reduced dimension
                #save to file
                os.makedirs(os.path.dirname(feature_file), exist_ok=True)
                np.save(feature_file, feat_mtx_dict[feat_name])

            else:
                feat_mtx_dict[feat_name] = np.load(feature_file)
                feat_dim_dict[feat_name] = feat_mtx_dict[feat_name].shape[1]


    return feat_mtx_dict, feat_dim_dict

