#!/bin/sh
### General options
#BSUB -q gpuv100
#BSUB -J M_influence_eval
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"
#BSUB -W 04:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -u s243425@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o ./logs/gpu_%J.out
#BSUB -e ./logs/gpu_%J.err

module purge
module load cuda/12.1

source ../.venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

# Path to the experiment directory generated with mc_size=5
EXP_PATH="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234"

# Evaluate entropy for M = 1, 2, 3, 4, 5 reusing cached images
python ../evaluate_M_influence.py \
    --path "$EXP_PATH" \
    --M_max 5 \
    --M_values 1 2 3 4 5
