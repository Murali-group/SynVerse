#!/bin/bash
#SBATCH --job-name=SynVerse_MLP
#SBATCH -p a100_dev_q
#SBATCH --output=%j-%a.o
#SBATCH --error=%j-%a.e
#SBATCH --array=1-2
#SBATCH -t 0-1:59:00 # time max for v100_dev_q its 2 hour
#SBATCH --mem=180G
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --account=synverse # give the account /project here
#SBATCH --export=NONE # this makes sure the compute environment is clean
#SBATCH --ntasks=2

# enter the virtual environment
module load Anaconda3/2020.11
cd /home/tasnina/Projects/SynVerse/code/
source activate synergy
echo $PWD
echo "Running for config file: "
echo $1
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
   CUDA_LAUNCH_BLOCKING=1 python -u main.py --config $1 --run_id $SLURM_JOB_ID --nic_name eno1 --shared_directory .
else
  CUDA_LAUNCH_BLOCKING=1 python -u main.py --config $1 --run_id $SLURM_JOB_ID --nic_name eno1 --shared_directory . --worker
fi



