#!/bin/bash
# ============================================================================
# Bash script to submit multiple jobs


# Optional: message to put into the created directory? TODO fix this
dir_message="2 arms task with hallway lengths 8, 64, 512 and 1024, AC agent"

# The parent experimental directory, each experiment will be stored as a sub-
# directory within this directory
dir_path="/network/tmp1/chenant/ant/cred_transport/long_arms/07-02/exp1_ac"

# Path to the parent directory containing the configuration files, one per
# experiment. Experiment folder will be named after the config file names
exp_config_dir_path="$dir_path/config"

# Job file path
job_file="/home/mila/c/chenant/repos/credit-transport/long_arms/arg-job_train-longarm.script"

# Job partition (same for all jobs)
partition_per_job="main,long"

# Job resource (same for all jobs)
gres_per_job="gpu:1"

# Specify cpu need (same for all jobs)
cpu_per_task="1"

# Specify memory (RAM) need (same for all jobs)
mem_per_job="12G"

# Specify time need (same for all jobs)
time_per_job="12:00:00"




# ============================================================================
# Below is automatically ran

# ==
# Little print-out
echo "Job submitted: `date`" > "$dir_path/submit_note.txt"
echo $dir_message >> "$dir_path/submit_note.txt"

# ==
# Iterate over each configuration file
for config_file in $exp_config_dir_path/*.ini ; do
  # Get the config name without parent path or file suffix
  exp_name=$(basename --suffix='.ini' "$config_file")
  exp_path="$dir_path/$exp_name"

  # Create exp directory
  echo "Experiment directory path: $exp_path"
  if [ ! -d "$exp_path" ] ; then
    mkdir -p $exp_path
  fi

  # Create error and output files
  cur_error_file="$exp_path/error_$exp_name.txt"
  cur_out_file="$exp_path/out_$exp_name.txt"

  # ==
  # Submit job
  sbatch --cpus-per-task=$cpu_per_task \
         --partition=$partition_per_job \
         --gres=$gres_per_job \
         --mem=$mem_per_job \
         --time=$time_per_job \
         --output="$cur_out_file" \
         --error="$cur_error_file" \
         --export=logpath="$exp_path",configpath="$config_file" \
         --job-name=$exp_name \
         $job_file

done



