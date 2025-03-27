#!/bin/bash
#SBATCH --job-name=SynVerse
#SBATCH -p a100_normal_q
#SBATCH --output=%j-%a.o
#SBATCH --error=%j-%a.e
#SBATCH --array=1-16
#SBATCH -t 0-36:00:00
#SBATCH --mem=100G
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --account=synverse # give the account /project here
#SBATCH --export=NONE # this makes sure the compute environment is clean
#SBATCH --ntasks=12


# enter the virtual environment
module load Anaconda3/2020.11
cd /home/tasnina/Projects/SynVerse/code/
source activate synergy
echo $PWD
echo "Running for config file: "
echo $1
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
  # Check if the third argument is provided
  if [ -n "$3" ]; then
    CUDA_LAUNCH_BLOCKING=1 python -u main.py --config $1 --feat "$2" --run_id $SLURM_JOB_ID --nic_name eno1 --shared_directory . > "$3" 2>&1
  else
    CUDA_LAUNCH_BLOCKING=1 python -u main.py --config $1 --run_id $SLURM_JOB_ID --nic_name eno1 --shared_directory . > "$2" 2>&1
  fi

else
   # Check if the third argument is provided
  if [ -n "$3" ]; then
    CUDA_LAUNCH_BLOCKING=1 python -u main.py --config $1 --feat "$2" --run_id $SLURM_JOB_ID --nic_name eno1 --shared_directory . --worker > "$3" 2>&1
  else
    CUDA_LAUNCH_BLOCKING=1 python -u main.py --config $1 --run_id $SLURM_JOB_ID --nic_name eno1 --shared_directory . --worker > "$2" 2>&1
  fi
fi




