import pandas as pd
import numpy as np
from transformers import BertTokenizer, WordpieceTokenizer

#***************************************************** FEATURE PREP ************************

def get_vocab_smiles(smiles_df):
    # vocabulary of SMILES characters
    vocab = sorted(set(''.join(smiles_df['smiles'].values))) #may consider using other tokenization

    # add special tokens
    char_to_token = {'[PAD]': 0, '[CLS]': 1}
    # SMILES tokens start from 2
    char_to_token.update({char: idx + 2 for idx, char in enumerate(vocab)})

    def tokenize(smiles, char_to_token):
        # prepend [CLS] token to each SMILES string
        return [char_to_token['[CLS]']] + [char_to_token[char] for char in smiles]

    smiles_df['tokenized'] = smiles_df['smiles'].apply(lambda x: tokenize(x, char_to_token))

    return smiles_df, len(char_to_token)
