#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J adm_diffusion_test
### -- ask for number of cores (must be at least 4 for GPU jobs) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- Request a GPU with 32GB VRAM (Crucial for your memory error) --
#BSUB -R "select[gpu32gb]"
### -- set walltime limit: hh:mm --
#BSUB -W 03:00
#BSUB -R "rusage[mem=24GB]"
### -- set the email address --
#BSUB -u s243425@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o ./logs/gpu_%J.out
#BSUB -e ./logs/gpu_%J.err
# -- end of LSF options --

# 1. Load Modules (Exact versions might vary, these are standard for DTU)
module purge
module load cuda/12.1

# 2. Activate Virtual Environment
# Assuming your .venv is one level up from ADM folder
source ../.venv/bin/activate

base="adm_full_script"

source "$base/vars.sh"
echo "EXP_PATH: $EXP_PATH"
echo "DATA_PATH: $DATA_PATH"

# 3. Memory Fragmentation Fix (Helps with OOM errors)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. Debug: Print GPU info to log
nvidia-smi

# 5. Run the script
# We run main.sh directly. Ensure main.sh is executable (chmod +x main.sh)
# bash ./$base/1_dataset.sh           # one-time: package val images as zip
# bash ./$base/2_fid_ref_stats.sh     # one-time: compute FID reference stats and real-image P&R features

# Extract Inception features for ALL generated images (prerequisite for realism scoring).
bash ./$base/3_all_features.sh

# Compute per-image realism scores from the full feature set.
bash ./$base/3_realism_eval.sh

# Run all three baselines (Random / GU / Realism) at each budget N.
for N in 12000 11000 10000 9000 8000 7000 6000 
do
  bash ./$base/4_baselines.sh $N
done