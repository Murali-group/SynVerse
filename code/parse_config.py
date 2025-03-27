from dataclasses import dataclass
from typing import Dict, Any
from typing import Optional

import types
@dataclass
class Inputs:
    synergy_file: str
    maccs_file: Optional[str] = None
    mfp_file: Optional[str] = None
    ecfp_file: Optional[str] = None
    smiles_file: Optional[str] = None
    mol_graph_file: Optional[str] = None
    target_file: Optional[str] = None
    genex_file: Optional[str] = None
    lincs: Optional[str] = None
    net_file: Optional[str] = None
    prot_info_file: Optional[str] = None

@dataclass
class Params:
    score_name: str
    drug_features: Any
    cell_line_features: Any
    model_info: Any
    epochs: int
    autoencoder_dims: Any
    splits: Any
    abundance: float
    max_drug_feat: int
    max_cell_feat: int
    min_drug_feat: int
    min_cell_feat: int
    hp_tune: bool
    train_mode: Any
    rewire_method: str
    shuffle: bool
    batch_size: int
    wandb: Any
    bohb: Any
    input_dir: str
    out_dir: str
    split_dir: str


def parse_config(config_map, **kwargs):
    input_settings = config_map['input_settings']
    input_dir = input_settings['input_dir']

    # Create Inputs dataclass
    inputs = Inputs(
        **{key: f"{input_dir}/{value}" for key, value in input_settings['input_files'].items()}
    )

    # Create Params dataclass
    params = Params(
        score_name= input_settings['score_name'],
        drug_features=input_settings['drug_features'],
        cell_line_features=input_settings['cell_line_features'],
        model_info=input_settings['model_info'],

        epochs=input_settings.get('epochs', 1500),
        batch_size=input_settings.get('batch_size', 4096),
        autoencoder_dims=input_settings.get('autoencoder_dims',  [[1024, 512], [512, 256], [256, 128], [256, 64]]),
        splits=input_settings['splits'],

        abundance=input_settings.get('abundance', 0.05),
        max_drug_feat=input_settings.get('max_drug_feat', 1),
        max_cell_feat=input_settings.get('max_cell_feat', 1),
        min_drug_feat=input_settings.get('min_drug_feat', 1),
        min_cell_feat=input_settings.get('min_cell_feat', 1),

        hp_tune=input_settings['hp_tune'],
        train_mode=input_settings['train_mode'],
        rewire_method=input_settings.get('rewire_method', None),
        shuffle=input_settings.get('shuffle', False),
        wandb = input_settings.get('wandb', {}),
        bohb=input_settings.get('bohb',{}),

        input_dir=input_dir,
        out_dir=config_map['output_settings']['output_dir'],
        split_dir=f"{input_dir}/splits",
    )

    #do some checks here.
    #if hp_tune==True then bohb cannot be empty dict
    if params.hp_tune:
        assert params.bohb!={}, print('BOHB param required in config for hyperparameter tuning')

    #if wandb is enabled then entity name and token in required.
    if params.wandb.get('enabled', False):
        assert params.wandb.get('entity_name', None)!=None, print('Entity name param required in config for wandb')
        assert params.wandb.get('token', None)!=None, print('Token param required in config for wandb')

    #If some params are overlapping across config_map and kwargs, then prioritize kwargs.
    override_params(params, **kwargs)

    #make sure that if some feature is present in config then corresponding input file is also available.

    return inputs, params

def override_params(params, **kwargs):
    #override directive about splits in params with directive from kwargs.get('split') if present.
    overriding_splits = kwargs.get('split')
    if overriding_splits:
        params.splits = []
        for overriding_split in overriding_splits.split(' '):
            split_dict = {'type': overriding_split, 'test_frac':0.2, 'val_frac':0.25}
            params.splits.append(split_dict)


    #override directive about which features to use in params with directive from kwargs.get('feat') if present.
    #run the model with the given exact combination of the features in kwargs.get('feat').
    # Note: the feature has to be present in the config file so that the other directives about feature preprocessing
    # and encoding is known.
    # (preprocess, encoder, norm etc.)
    feat_to_use = kwargs.get('feat')
    if feat_to_use is not None:
        use_feat = feat_to_use.split(' ')
        for feat in params.drug_features:
            if feat['name'] in use_feat:
                feat['use'] = [True]
            else:
                feat['use'] = [False]

        for feat in params.cell_line_features:
            if feat['name'] in use_feat:
                feat['use'] = [True]
            else:
                feat['use'] = [False]
    return params




