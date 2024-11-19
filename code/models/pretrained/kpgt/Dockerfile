ARG IMAGE_TAG=23.09-py3

FROM nvcr.io/nvidia/tensorrt:${IMAGE_TAG} as base

FROM base as training

# Install required dependencies for the conda environment
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    make gcc gcc-multilib \
    daemontools \
    openssh-server \
    git-all \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda for environment management
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean -ya

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# install training requirements
COPY environment.yml environment.yml

# Create the conda environment
RUN conda env create -f environment.yml && \
    conda clean -a -y
RUN conda env list && conda run -n kpgt python -c "import torch; print(torch.__version__)"

# Set the environment to use the conda environment
SHELL ["conda", "run", "-n", "kpgt", "/bin/bash", "-c"]

# Activate the conda environment explicitly for future commands
ENV PATH /opt/conda/envs/kpgt/bin:$PATH

# Clone the repository and install any additional dependencies
RUN git clone https://github.com/lihan97/KPGT.git && \
    cd KPGT