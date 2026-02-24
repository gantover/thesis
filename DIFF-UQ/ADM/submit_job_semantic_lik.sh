#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J semantic_likelyhood_eval 
### -- ask for number of cores (must be at least 4 for GPU jobs) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- Request a GPU with 32GB VRAM (Crucial for your memory error) --
#BSUB -R "select[gpu32gb]"
### -- set walltime limit: hh:mm --
#BSUB -W 04:00
### -- request 16GB of system RAM --
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
#BSUB -u s243425@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

# 1. Load Modules (Exact versions might vary, these are standard for DTU)
module purge
module load cuda/12.1

# 2. Activate Virtual Environment
# Assuming your .venv is one level up from ADM folder
source ../.venv/bin/activate

# 3. Memory Fragmentation Fix (Helps with OOM errors)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. Debug: Print GPU info to log
nvidia-smi

# 5. Run the script
python ../semantic_likelihood.py --path "/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234"