from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from dgl.data.utils import load_graphs
import torch
import dgl.backend as F
import scipy.sparse as sps

SPLIT_TO_ID = {'train':0, 'val':1, 'test':2}
class MoleculeDataset(Dataset):
    def __init__(self, root_path, dataset, use_idxs, path_length=5):
        dataset_path = os.path.join(root_path, f"{dataset}/{dataset}.csv")
        self.cache_path = os.path.join(root_path, f"{dataset}/{dataset}_{path_length}.pkl")

        ecfp_path = os.path.join(root_path, f"{dataset}/rdkfp1-7_512.npz")
        md_path = os.path.join(root_path, f"{dataset}/molecular_descriptors.npz")
        # Load Data
        df = pd.read_csv(dataset_path)

        fps = torch.from_numpy(sps.load_npz(ecfp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))
        self.df, self.fps, self.mds = df.iloc[use_idxs], fps[use_idxs], mds[use_idxs]
        self.smiless = self.df['smiles'].tolist()
        self.use_idxs = use_idxs
        # Dataset Setting
        self.task_names = self.df.columns.drop(['smiles']).tolist()
        self.n_tasks = len(self.task_names)
        self._pre_process()
        self.mean = None
        self.std = None
        self.set_mean_and_std()
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]
    def _pre_process(self):
        if not os.path.exists(self.cache_path):
            print(f"{self.cache_path} not exists, please run preprocess.py")
        else:
            graphs, label_dict = load_graphs(self.cache_path)

            self.graphs = []
            for i in self.use_idxs:
                self.graphs.append(graphs[i])
            self.labels = label_dict['labels'][self.use_idxs]
        self.fps, self.mds = self.fps,self.mds
    def __len__(self):
        return len(self.smiless)
    
    def __getitem__(self, idx):
        return self.smiless[idx], self.graphs[idx], self.fps[idx], self.mds[idx], self.labels[idx]

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std


