#!/bin/bash
#SBATCH -J SynVerse #(run name give any name you want to track)
#SBATCH --account=synverse # give the account /project here
#SBATCH -p dgx_normal_q # partition a100_normal_q takes time to get resource so test your code with v100_dev_q
#SBATCH -N 1  # this requests 1 node
#SBATCH -t 0-96:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --export=NONE # this makes sure the compute environment is clean

module load Anaconda3/2020.11
cd /home/tasnina/Projects/SynVerse/code/
source activate synergy
echo "hello world"
echo $PWD

# Check if the third argument is provided
if [ -n "$3" ]; then
    CUDA_LAUNCH_BLOCKING=1 python -u main.py --config "$1" --feat "$2" --run_id "$SLURM_JOB_ID" > "$3" 2>&1
elif [ -n "$5" ]; then
    CUDA_LAUNCH_BLOCKING=1 python -u main.py --config "$1" --feat "$2" --start_run "$3" --end_run "$4" --run_id "$SLURM_JOB_ID" > "$5" 2>&1
else
    CUDA_LAUNCH_BLOCKING=1 python -u main.py --config "$1" --run_id "$SLURM_JOB_ID" > "$2" 2>&1
fi

