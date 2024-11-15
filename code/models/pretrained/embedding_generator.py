
def get_SPMM_embedding(smiles, input_dir, device):
    from models.pretrained.SPMM.encoder import SPMM_Encoder

    vocab_file = input_dir + 'drug/vocab_bpe_300.txt'
    checkpoint_file = input_dir + 'drug/pretrain/checkpoint_SPMM.ckpt'

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

def get_MolE_embedding(smiles, input_dir, device):
    from models.pretrained.mole_public.mole.cli import mole_predict
    checkpoint_file = input_dir + 'drug/pretrain/MolE_GuacaMol_27113.ckpt'
    embeddings = mole_predict.encode(smiles=smiles, pretrained_model= checkpoint_file, batch_size = 32, num_workers = 4)

    return embeddings, embeddings.shape[1]






