#!/bin/bash
#SBATCH -J SynVerse_MLP #(run name give any name you want to track)
#SBATCH -p a100_dev_q # partition a100_normal_q takes time to get resource so test your code with v100_dev_q 
#SBATCH -N 1  # this requests 1 node, 1 core.
#SBATCH --ntasks=12
#SBATCH -t 0-1:59:00 # time max for v100_dev_q its 2 hour
#SBATCH --mem=180G
#SBATCH --exclusive
#SBATCH --gres=gpu:1 
#SBATCH --account=synverse # give the account /project here 
#SBATCH --export=NONE # this makes sure the compute environment is clean

module load Anaconda3/2020.11
cd /home/tasnina/Projects/SynVerse/code/
source activate synergy
echo "hello world"
echo $PWD
CUDA_LAUNCH_BLOCKING=1 python main.py --config "/home/tasnina/Projects/SynVerse/code/config_files/arc_master_config.yaml"