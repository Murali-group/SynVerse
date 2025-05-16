# SynVerse
## Table of Contents
- [Introduction](#introduction)
- [Conda Environment Setup](#conda-environment-setup)
- [Download Processed Dataset](#download-processed-dataset)
- [How to Use SynVerse](#how-to-use-synverse)
    - [Configuration File](#configuration-file)
- [Docker Setup Guide for Pretrained Models](#docker-setup-guide-for-pretrained-models)
  - [Prerequisites](#prerequisites)
  - [Build the Docker Image](#build-the-docker-image)
- [Publication](#publication)
      
## Introduction
SynVerse is a framework with an encoder-decoder architecture. It incorporates diverse input features and a reasonable approximation of model architectures commonly employed by existing deep learning-based synergy prediction methods. It includes four data-splitting strategies and three ablation methods: module-based, feature shuffling, and a novel network-based approach to isolate factors influencing model performance.
<div align="center">
    <img src="https://github.com/Murali-group/SynVerse/blob/main/SynVerse_Overview.jpg" alt="Screenshot" style="width: 70%;">
</div>

## Conda Environment Setup
If you haven't cloned the repository yet, run the following command to clone it and navigate to the SynVerse folder:
```bash
git clone https://github.com/Murali-group/SynVerse.git
cd SynVerse
```

Then, follow the steps below to set up the `synverse` environment with required libraries using the provided [`synverse.yml`](./synverse.yml) file.

```bash
conda env create -f synverse.yml
```
To run the command, make sure Conda is installed. If not, install [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) or the lighter version, [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install).

After the environment is created, activate it using:
```bash
conda activate synverse
```

To verify that the environment and its dependencies are set up correctly, you can list the installed packages:
```bash
conda list
```

## Download Processed Dataset
All datasets used in this study, including drug and cell line features and the preprocessed synergy dataset required to reproduce the results, are available in the [Zenodo repository](https://zenodo.org/records/15277144). Download and unzip the `inputs.zip` file, and place the `inputs` folder in the project directory, at the same level as the `code/` folder.


## How to Use SynVerse

SynVerse is configured using a YAML file (e.g., [sample_config.yaml](https://github.com/Murali-group/SynVerse/blob/main/code/config_files/sample_config.yaml)), which allows users to define the input features, model architecture, and evaluation strategies. Once a configuration file is prepared, SynVerse can be run in various modes to perform different tasks:

1. To train and test a model:
```
   python main.py --config_file config_files/sample_config.yaml --train_type 'regular'
```
2. To perform feature-shuffling-based ablation study:
```
   python main.py --config_file config_files/sample_config.yaml --train_type 'shuffle'
```
3. To perform network-based ablation study:
```
   python main.py --config_file config_files/sample_config.yaml --train_type 'rewire'
```


### Configuration File

This section describes each field in the YAML configuration file used by SynVerse. 

####  `score_name`
The synergy score to predict (Options:`'S_mean_mean'`, `'synergy_loewe_mean'`)

####  `input_dir`
Base directory where all input files are stored.

####  `input_files`
Each entry defines a path to a required input file.
| Key | Description |
|-----|-------------|
| `synergy_file` | Contains synergy triplets. Required columns: `drug_1_pid`, `drug_2_pid`, `cell_line_name`, and `S_mean_mean` (or `synergy_loewe_mean`). |
| `maccs_file` | MACCS fingerprint file. Columns: `pid`, `MACCS_0`, ..., `MACCS_166`. |
| `mfp_file` | Morgan fingerprints. Columns: `pid`, `Morgan_FP_0`, ..., `Morgan_FP_255`. |
| `ecfp_file` | ECFP_4 fingerprints. Columns: `pid`, `ECFP4_0`, ..., `ECFP4_1023`. |
| `smiles_file` | SMILES strings. Columns: `pid`, `smiles`. |
| `mol_graph_file` | Pickle file with DeepChem-derived molecular graphs: `{pid: graph}`. |
| `target_file` | Drug target binary profile. Columns: `pid` and target names |
| `genex_file` | Cell line gene expression. Columns: `cell_line_name` and gene names. |
| `lincs` | Landmark genes file for LINCS1000. |
| `vocab_file` | Vocabulary file to convert smiles to tokens. |
| `net_file` | STRING network file (gzipped). |
| `prot_info_file` | STRING protein metadata file (gzipped). |

---

####  `drug_features`

Describes drug-level features to be used.
- `name`: str: Feature name 
- `preprocess`: str: Preprocessing method 
- `compress`: bool: Use autoencoder to reduce dimensions.
- `norm`: str: Normalization method 
- `encoder`: str: Feature-specific encoders 
- `use`: List of boolean values: Determines if the feature should be used in the model.

####  `cell_line_features`

Same structure as `drug_features`, but for cell lines.

####  `model_info`

#####  `decoder`
- `name`: Model architecture (e.g., `'MLP'`).
- `hp_range`: Hyperparameter search space for tuning.
- `hp`: Default configuration.
  
#####  `drug_encoder`
List of encoder configs.
Each contains:
- `name`: Encoder type (e.g., `'GCN'`, `'Transformer'`).
- `hp_range`: Hyperparameter search space for tuning.
- `hp`: Default configuration.

####  `autoencoder_dims`
Dimension of hidden layers for the autoencoder.

####  `batch_size`
Number of samples per batch during training.

####  `epochs`
Maximum training epochs.

####  `splits`
- `type`: Spitting strategy to use (Options: `random`, `leave_comb`, `leave_drug`, `leave_cell_line`). 
- `test_frac`: Test set size (fraction of total).
- `val_frac`: Validation set size (fraction of training set).

#### `wandb`

[Weights & Biases](https://wandb.ai) integration for experiment tracking.
- `enabled`: Enable logging.
- `entity_name`, `token`, `project_name`: W&B credentials.
- `timezone`, `timezone_format`: Time info formatting.

####  `abundance`
Minimum percentage of triplets required per cell line to be included in training.

####  Feature Combination Control

Defines the number of features in combinations.

| Field | Meaning |
|-------|---------|
| `max_drug_feat` | Maximum number of drug features per model. |
| `min_drug_feat` | Minimum number of drug features per model. |
| `max_cell_feat` | Maximum number of cell line features per model. |
| `min_cell_feat` | Minimum number of cell line features per model. |

####  `hp_tune`

Set to `true` to enable hyperparameter optimization.

####  `bohb`
Settings for Bayesian Optimization with BOHB.
- `min_budget`, `max_budget`: Resource limits per trial.
- `n_iterations`: Number of trials.
- `run_id`: BOHB session ID.
- `server_type`: `'local'` or `'cluster'`.

####  `rewire_method`
Which network rewiring method to use when flag --train_type =  `rewire`.
Options: `SM`: Degree-preserving (Maslov-Sneppen), `SA`: Strength-preserving (Simulated Annealing).

####  `output_settings`

- `output_dir`: Output directory for results.

## Docker Setup Guide for Pretrained Models
This guide provides detailed instructions for containerizing [KPGT](https://github.com/lihan97/KPGT) and [MolE](https://github.com/recursionpharma/mole_public) using Docker with GPU support.

### Prerequisites
- **Hardware Requirements**
  - NVIDIA GPU with Compute Capability ≥ 3.5
  - Minimum 16GB+ RAM
  - 50GB+ free disk space for Docker images and dependencies

- **Software Requirements**
  - **For KPGT (CUDA 11.3)** 
    - **OS:** Linux (Ubuntu 20.04 recommended)
    - **NVIDIA drivers**: v465+ (CUDA 11.3 compatibility)
    - **CUDA Toolkit:** 11.3.1
    - **Docker:** Engine 20.10+ with NVIDIA Container Toolkit configured for CUDA 11.3
  - **For MolE (CUDA 12.1)** 
    - **OS:** Linux (Ubuntu 20.04 recommended)
    - **NVIDIA drivers**: v525+ (CUDA 12.1 compatibility)
    - **CUDA Toolkit:** 12.1
    - **Docker:** Engine 20.10+ with NVIDIA Container Toolkit configured for CUDA 12.1

### Build the Docker Image
We already have created Dockerfiles for each project. To build the Docker images, simply run the provided [script](./code/build_docker.sh).
This script will build the `kpgt:base` image and the `mole:base` image using their respective Dockerfiles located in the root directory of each project at [pretrain](./code/preprocessing/pretrain) directory.

## Publication
Tasnina, N., Haghani, M. and Murali, T.M., 2025. SynVerse: A Framework for Systematic Evaluation of Deep Learning Based Drug Synergy Prediction Models. bioRxiv, pp.2025-04.
https://doi.org/10.1101/2025.04.30.651516
