#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J adm_diffusion_test
### -- ask for number of cores (must be at least 4 for GPU jobs) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- Request a GPU with 40GB VRAM --
#BSUB -R "select[gpu40gb]"
### -- set walltime limit: hh:mm --
#BSUB -W 08:00
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
source ../../.venv/bin/activate

base="adm_full_script"

source "$base/vars.sh"
echo "EXP_PATH: $EXP_PATH"
echo "DATA_PATH: $DATA_PATH"

# Keep Hugging Face and temporary files on BLACKHOLE instead of HOME quota.
export HF_HOME="${REF_PATH}/_hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_XET_CACHE="${HF_HOME}/xet"
export TMPDIR="${REF_PATH}/_tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TMPDIR"
echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "HF_XET_CACHE: $HF_XET_CACHE"
echo "TMPDIR: $TMPDIR"

# If a prior V100 run patched HPSv3 config, restore the original for A100 jobs.
# HPSV3_DEFAULT_CFG="../../.venv/lib64/python3.9/site-packages/hpsv3/config/HPSv3_7B.yaml"
# if [ -f "${HPSV3_DEFAULT_CFG}.bak" ]; then
# 	cp "${HPSV3_DEFAULT_CFG}.bak" "$HPSV3_DEFAULT_CFG"
# 	echo "Restored original HPSv3 config: $HPSV3_DEFAULT_CFG"
# fi

# 3. Memory Fragmentation Fix (Helps with OOM errors)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. Debug: Print GPU info to log
nvidia-smi
export CUDA_VISIBLE_DEVICES=0
nvidia-smi

# bash ./$base/5_hpsv3_score.sh
bash ./$base/5_hpsv3_score_classes_npy.sh