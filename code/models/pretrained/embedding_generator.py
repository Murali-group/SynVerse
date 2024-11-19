import os.path
import sys
import subprocess
import json
import numpy as np
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
    return np.array(embeddings), np.array(embeddings).shape[1]

def get_mole_embedding(smiles, input_dir):
    # Path to the checkpoint file
    local_dir = f'{input_dir}/drug/pretrain/'
    docker_dir = "/mnt"
    ckpt_file = f'{docker_dir}/MolE_GuacaMol_27113.ckpt'

    embeddings = []
    b_size = 512
    i = 0
    while True:
        if i >= len(smiles):
            break
        smiles_batch = smiles[i:min(i + b_size, len(smiles))]
        i=i+b_size

        # List of SMILES strings
        smiles_str = repr(smiles_batch)
        docker_command = [
            "docker", "run",
            "-v", f"{local_dir}:{docker_dir}",  # Mount local directory to docker
            "mole:base",  # The Docker image name
            "python", "-c",  # Run Python code directly
            f"from mole_public.mole.cli.mole_predict import encode; "
            f"import json; "
            f"embeddings = encode(smiles={smiles_str}, pretrained_model='{ckpt_file}', batch_size=32, num_workers=4); "
            f"print('EMBEDDINGS:'); "
            f"print(json.dumps(embeddings.tolist()))"
        ]

        # Run the Docker command and capture the output (embeddings)
        result = subprocess.run(docker_command, capture_output=True, text=True, check=True)

        # Read the embeddings from the output file
        embedding_list = result.stdout.split('EMBEDDINGS:\n')[-1]
        embeddings.extend((eval(embedding_list)))


    return np.array(embeddings),np.array(embeddings).shape[1]

