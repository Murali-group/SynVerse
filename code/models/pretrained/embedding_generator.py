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

"""Helper function to run a docker command and handle errors."""
def run_docker_command(command, success_message):
    try:
        subprocess.run(command, check=True)
        print(success_message)
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")

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

    mole_embed_file = f'{input_dir}/drug/mole_embed.npy'

    if not (os.path.exists(mole_embed_file)):
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

        mole_embed = np.array(embeddings)
        #save embeddings
        os.makedirs(os.path.dirname(mole_embed_file), exist_ok=True)
        np.save(mole_embed_file, mole_embed)

    mole_embed = np.load(mole_embed_file)
    return mole_embed, mole_embed.shape[1]

# Preprocess SMILES and get molecular graph
def kpgt_process_smiles(smiles, input_dir):
    # prepare the SMILES CSV file
    host_data_path = os.path.join(input_dir, "drug", "pretrain")
    kpgt_prefix = "kpgt_smiles"
    smiles_dir = os.path.join(host_data_path, kpgt_prefix)
    os.makedirs(smiles_dir, exist_ok=True)
    smiles_file_path = os.path.join(smiles_dir, f"{kpgt_prefix}.csv")
    pd.DataFrame({'smiles': smiles}).to_csv(smiles_file_path, index=False)
    pretrain_dir = os.path.join(host_data_path, kpgt_prefix)
    docker_image = "kpgt:base"
    container_data_path = "/app/datasets"  # path inside the container

    preprocessed_smiles_file_path = os.path.join(pretrain_dir, "molecular_descriptors.npz")
    if not os.path.exists(preprocessed_smiles_file_path):
        preprocess_script = "preprocess_downstream_dataset.py"
        preprocess_command = [
            "docker", "run", "--rm",
            "-v", f"{host_data_path}:{container_data_path}",
            "-w", "/workspace/KPGT/scripts",
            docker_image,
            "python", preprocess_script,
            "--data_path", container_data_path,
            "--dataset", kpgt_prefix
        ]
        run_docker_command(preprocess_command, "Data preprocessing finished successfully.")

    # # Load molecular_descriptor npz file which includes 'md' key (a 200 dimension representation for each SMILES)
    # mds = np.load(preprocessed_smiles_file_path)['md'].astype(np.float32)
    # mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))

    # ecfp_path = os.path.join(pretrain_dir, "rdkfp1-7_512.npz")
    # fps = torch.from_numpy(sps.load_npz(ecfp_path).todense().astype(np.float32))

    # return mds, fps,  mds.shape[1], fps.shape[1]


# Generate embedding
def get_kpgt_embedding(smiles, input_dir, device=None):
    kpgt_process_smiles(smiles, input_dir)

    host_data_path = os.path.join(input_dir, "drug", "pretrain")  # path to dataset directory on the host
    kpgt_prefix = "kpgt_smiles"
    config = "base"
    pretrain_dir = os.path.join(host_data_path, kpgt_prefix)
    docker_image = f"kpgt:{config}"
    container_data_path = "/app/datasets"  # path inside the container
    kpgt_embed_file_path = os.path.join(pretrain_dir, f"kpgt_{config}.npz")

    if not os.path.exists(kpgt_embed_file_path):
        embed_script = "extract_features.py"
        model_path = os.path.join(container_data_path, "pretrained_kpgt.pth")
        embed_command = [
            "docker", "run", "--rm",
            "--gpus", "all",
            "--runtime", "nvidia",
            "-v", f"{host_data_path}:{container_data_path}",
            "-w", "/workspace/KPGT/scripts",
            docker_image,
            "python", embed_script,
            "--config", "base",
            "--data_path", container_data_path,
            "--dataset", kpgt_prefix,
            "--model_path", model_path
        ]
        run_docker_command(embed_command, f"Embedding generation finished successfully.")

    kpgt_embed = np.load(kpgt_embed_file_path)['fps']
    return kpgt_embed, kpgt_embed.shape[1]
