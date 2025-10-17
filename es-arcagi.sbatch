#!/bin/bash                     
#SBATCH --job-name=es-arcagi  # Name of the job
#SBATCH --output=/orcd/data/tpoggio/001/tiffany8/es-urop/es-arcagi.out
#SBATCH --error=/orcd/data/tpoggio/001/tiffany8/es-urop/es-arcagi.err
#SBATCH -n 8                     # 8 CPU core
#SBATCH --gres=gpu:h100:4            # 4 H100 GPU 
#SBATCH --mem=64G
#SBATCH --time=00:05:00               # Time limit (HH:MM:SS)
#SBATCH --signal=B:USR1@120
#SBATCH --partition=ou_bcs_normal

set -x

whoami

source ~/.bashrc
source /home/tiffany8/miniconda3/etc/profile.d/conda.sh
conda activate myenv 

echo "check gpu status before running job"
nvidia-smi

cd /orcd/data/tpoggio/001/tiffany8/es-urop

export PROJECT_ROOT=/orcd/data/tpoggio/001/tiffany8/es-urop
export MODEL_NAME=/orcd/data/tpoggio/001/om/tiffany8/grpo-urop/TinyZero/model/Qwen2.5-3B-Instruct/
export SCRATCH_DIR=/home/tiffany8/orcd/scratch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

accelerate launch \
  --num_processes 4 \
  /orcd/data/tpoggio/001/tiffany8/es-urop/es_fine-tuning_arcagi.py \
  --model_name "$MODEL_NAME" \
  --project_path "$PROJECT_ROOT" \
  --hf_cache_dir "$SCRATCH_DIR" \
  --mixed_precision "bf16" \
  --verbose \
  --log_sample_every 1
