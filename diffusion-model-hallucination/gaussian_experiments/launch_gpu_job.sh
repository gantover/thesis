#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J Toy_Dataset_Diffusion_Hallucination_Exp
### -- ask for number of cores (must be at least 4 for GPU jobs) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"
### -- set walltime limit: hh:mm --
#BSUB -W 08:00
### -- request 16GB of system RAM --
#BSUB -R "rusage[mem=16GB]"
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

module purge
module load cuda/12.1

# 2. Activate Virtual Environment
# Assuming your .venv is one level up from ADM folder
source ../../DIFF-UQ/.venv/bin/activate

# 3. Memory Fragmentation Fix (Helps with OOM errors)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. Debug: Print GPU info to log
nvidia-smi

# 5. Run the script
# bash ./jazbec_experiment/run.sh
# bash ./jazbec_experiment/run_quarter.sh
python jazbec_compute2.py