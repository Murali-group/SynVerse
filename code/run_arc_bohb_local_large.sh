#!/bin/bash
#SBATCH -J SynVerse #(run name give any name you want to track)
#SBATCH -p dgx_normal_q # partition a100_normal_q takes time to get resource so test your code with v100_dev_q
#SBATCH -N 1  # this requests 1 node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-24:00:00
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --account=synverse # give the account /project here
#SBATCH --export=NONE # this makes sure the compute environment is clean

module load Anaconda3/2020.11
cd /home/tasnina/Projects/SynVerse/code/
source activate synergy
echo "hello world"
echo $PWD

# Check if the third argument is provided
if [ -n "$3" ]; then
    CUDA_LAUNCH_BLOCKING=1 python -u main.py --config "$1" --feat "$2" > "$3" 2>&1
else
    CUDA_LAUNCH_BLOCKING=1 python -u main.py --config "$1" > "$2" 2>&1
fi

