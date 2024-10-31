import torch
import torch.nn as nn
import math
from models.encoders.drug.xbert import BertConfig, BertForMaskedLM
from transformers import BertTokenizer, WordpieceTokenizer


class SPMM_embedder(nn.Module):
    def __init__(self, tokenizer=None, config=None):
        super().__init__()
        self.tokenizer = tokenizer
        # bert_config = BertConfig.from_json_file(config['bert_config_text'])
        bert_config = BertConfig(**config)
        self.text_encoder = BertForMaskedLM(config=bert_config)
        for i in range(bert_config.fusion_layer, bert_config.num_hidden_layers):  self.text_encoder.bert.encoder.layer[i] = nn.Identity()
        self.text_encoder.cls = nn.Identity()

    def forward(self, text_input_ids, text_attention_mask):
        vl_embeddings = self.text_encoder.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
        vl_embeddings = vl_embeddings[:, 0, :]
        return vl_embeddings

class SPMM_Encoder(nn.Module):
    def __init__(self, vocab_file, checkpoint_file, config, device):
        super().__init__()
        self.device = device
        #TODO check if out_dim is right
        self.out_dim=config['embed_dim']

        self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=False, do_basic_tokenize=False)
        self.tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.tokenizer.vocab, unk_token=self.tokenizer.unk_token,
                                                           max_input_chars_per_word=250)
        self.model = SPMM_embedder(config=config, tokenizer=self.tokenizer)

        print(f'LOADING PRETRAINED MODEL from {checkpoint_file}')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        state_dict = checkpoint['state_dict']
        print('LOADING COMPLETE for PRETRAINED MODEL..')

        for key in list(state_dict.keys()):
            if '_unk' in key:
                new_key = key.replace('_unk', '_mask')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        print('load checkpoint from %s' % checkpoint_file)

    def forward(self,smiles):
        #add '[CLS]' token before smiles.
        text = ['[CLS]'+x for x in smiles]
        text_input = self.tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(self.device)
        embedding = self.model(text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:])
        return embedding
