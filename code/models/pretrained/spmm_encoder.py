import torch
import torch.nn as nn
import math
from xbert import BertConfig, BertForMaskedLM
from transformers import BertTokenizer, WordpieceTokenizer


class SPMM_embedder(nn.Module):
    def __init__(self, tokenizer=None, config=Nne):
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
    def __init__(self, vocab_file, checkpoint_file, device):
        super().__init__()
        #config for pretrained model in SPMM
        SPMM_config = {
            "architectures": ["BertForMaskedLM"],
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "embed_dim": 256,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 0.000000000001,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 300,
            "fusion_layer": 6,
            "encoder_width": 768,
            "autoregressive": 0,
            "add_cross_attention": "True"
            }

        # SPMM_config.update(config)

        self.device = device
        self.out_dim=SPMM_config['hidden_size']

        self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=False, do_basic_tokenize=False)
        self.tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.tokenizer.vocab, unk_token=self.tokenizer.unk_token,
                                                           max_input_chars_per_word=250)
        self.spmm_embedder = SPMM_embedder(config=SPMM_config, tokenizer=self.tokenizer)

        print(f'LOADING PRETRAINED MODEL from {checkpoint_file}')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        state_dict = checkpoint['state_dict']
        print('LOADING COMPLETE for PRETRAINED MODEL..')

        for key in list(state_dict.keys()):
            if '_unk' in key:
                new_key = key.replace('_unk', '_mask')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        self.spmm_embedder.load_state_dict(state_dict, strict=False)
        self.spmm_embedder.to(self.device)

        # # Freeze BERT model
        for param in self.spmm_embedder.parameters():
            param.requires_grad = False
        print('load checkpoint from %s' % checkpoint_file)

        # Nure: adding a layer norm after getting embedding from pretrained model.
        # self.layer_norm_1 = nn.LayerNorm(SPMM_config['hidden_size'])


    def forward(self,smiles):
        #add '[CLS]' token before smiles.
        text = ['[CLS]'+x for x in smiles]
        text_input = self.tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(self.device)
        embedding = self.spmm_embedder(text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:])
        # embedding = self.layer_norm_1(embedding)

        # print('embedding shape: ', embedding.shape)
        return embedding

def get_SPMM_embedding(smiles, vocab_file, checkpoint_file, device):
    pretrained_spmm = SPMM_Encoder(vocab_file, checkpoint_file, device)
    #get drug embedding in batches
    embeddings=[]
    b_size=512
    i=0
    while True:
        if i>=len(smiles):
            break
        embeddings.extend(pretrained_spmm(smiles[i:min(i+b_size, len(smiles))]).cpu().numpy())
        i=i+b_size
    return embeddings, pretrained_spmm.out_dim
