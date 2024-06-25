#!/bin/bash
#SBATCH -J SynVerse #(run name give any name you want to track)
#SBATCH -p dgx_normal_q # partision a100_normal_q takes time to get resource so test your code with v100_dev_q
#SBATCH -N 1  # this requests 1 node
#SBATCH --ntasks=1
#SBATCH -t 0-1:00:00 # time max for v100_dev_q its 2 hour
#SBATCH --mem=16G
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --account=synverse # give the account /project here
#SBATCH --export=NONE # this makes sure the compute environment is clean
#SBATCH --output=test_job.out
#SBATCH --error=test_job.err

module load Anaconda3/2020.11
cd /home/tasnina/Projects/SynVerse/code/
source activate synergy
echo "hello world"
echo $PWD
CUDA_LAUNCH_BLOCKING=1 python -u 'test_code/dataparallel_test.py'
