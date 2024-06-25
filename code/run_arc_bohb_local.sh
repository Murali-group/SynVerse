#!/bin/bash
#SBATCH -J SynVerse #(run name give any name you want to track)
#SBATCH -p a100_normal_q # partision a100_normal_q takes time to get resource so test your code with v100_dev_q
#SBATCH -N 1  # this requests 1 node
#SBATCH --ntasks=12
#SBATCH -t 0-143:00:00
#SBATCH --mem=180G
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --account=synverse # give the account /project here
#SBATCH --export=NONE # this makes sure the compute environment is clean

module load Anaconda3/2020.11
cd /home/tasnina/Projects/SynVerse/code/
source activate synergy
echo "hello world"
echo $PWD
CUDA_LAUNCH_BLOCKING=1 python -u main.py --config $1 > $2 2>&1
