#!/bin/bash
#SBATCH -J SynVerse #(run name give any name you want to track)
#SBATCH --account=synverse # give the account /project here
#SBATCH -p dgx_normal_q # partition a100_normal_q takes time to get resource so test your code with v100_dev_q
#SBATCH -N 1  # this requests 1 node
#SBATCH -t 0-143:00:00
#SBATCH --gres=gpu:1
#SBATCH --export=NONE # this makes sure the compute environment is clean

module load Anaconda3/2020.11
cd /home/tasnina/Projects/SynVerse/code/
source activate synergy
echo "hello world"
echo $PWD

# Process named arguments using a loop
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) config="$2"; shift ;;           # Capture value for --config
        --seed) seed="$2"; shift ;;           # Capture value for --seed
        --feat) feat="$2"; shift ;;               # Capture value for --feat
        --split) split="$2"; shift ;;             # Capture value for --split
        --force_split) force_split="$2"; shift ;;             # Capture value for --force_split
        --start_run) start_run="$2"; shift ;;     # Capture value for --start_run
        --end_run) end_run="$2"; shift ;;         # Capture value for --end_run
        --output) output="$2"; shift ;;           # Capture value for --output
        *) echo "Unknown parameter passed: $1"; exit 1 ;;  # Handle unknown arguments
    esac
    shift  # Move to the next argument
done

# Set a default value for output if not provided
output="${output:-default_output.log}"

# Build the command dynamically
command="CUDA_LAUNCH_BLOCKING=1 python -u main.py"
[ -n "$config" ] && command+=" --config \"$config\""
[ -n "$seed" ] && command+=" --seed \"$seed\""
[ -n "$feat" ] && command+=" --feat \"$feat\""
[ -n "$split" ] && command+=" --split \"$split\""
[ -n "$force_split" ] && command+=" --force_split \"$force_split\""
[ -n "$start_run" ] && command+=" --start_run \"$start_run\""
[ -n "$end_run" ] && command+=" --end_run \"$end_run\""
command+=" --run_id \"$SLURM_JOB_ID\""
[ -n "$output" ] && command+=" > \"$output\" 2>&1"

# Execute the command
eval $command