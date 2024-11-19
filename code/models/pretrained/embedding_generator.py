import os.path
import sys
import subprocess
import json
import numpy as np
import pandas as pd

def get_pretrained_embedding(smiles, input_dir,encoder_name, device):
    prtrained_model_map = {"SPMM": get_SPMM_embedding, "mole": get_mole_embedding, "kpgt": get_kpgt_embedding}
    embedding, embed_dim = prtrained_model_map[encoder_name](smiles, input_dir, device)
    return embedding, embed_dim

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

def get_mole_embedding(smiles, input_dir, device=None):
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

def get_kpgt_embedding(smiles, input_dir, device=None):
    #save smile to a .csv file which is an acceptable form by KPGT
    host_data_path = f"{input_dir}/drug/pretrain/" # Absolute path to your dataset directory on the host
    smiles_file_name ='temp_smiles'
    smiles_file_path = host_data_path + smiles_file_name+'/'+ smiles_file_name+'.csv'
    os.makedirs(os.path.dirname(smiles_file_path), exist_ok=True)
    pd.DataFrame({'smiles':smiles}).to_csv(smiles_file_path, index=False)


    docker_image = "kpgt:base"
    container_data_path = "/app/datasets"  # Path inside the container
    preprocess_script = "preprocess_downstream_dataset.py"
    embed_script = "extract_features.py"
    model_file = "pretrained_kpgt.pth"

    # # Construct the docker run command to preprocess SMILES and gte molecular graph data
    # command = [
    #     "docker", "run", "--rm",
    #     "-v", f"{host_data_path}:{container_data_path}",
    #     "-w", "/workspace/KPGT/scripts",
    #     docker_image,
    #     "python", preprocess_script,
    #     "--data_path", container_data_path,
    #     "--dataset", smiles_file_name
    # ]
    #
    # # Run the command
    # try:
    #     subprocess.run(command, check=True)
    #     print("Data preprocessing finished successfully.")
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during execution: {e}")

    # Construct the docker run command to generate embedding
    command = [
        "docker", "run", "--rm",
        "-v", f"{host_data_path}:{container_data_path}",
        "-w", "/workspace/KPGT/scripts",
        docker_image,
        "python", embed_script,
        "--config", "base",
        "--model_path", f"{container_data_path}/{model_file}"
        "--data_path", container_data_path,
        "--dataset", smiles_file_name
    ]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print("Embedding generation finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")

