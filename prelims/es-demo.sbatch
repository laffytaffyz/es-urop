#!/bin/bash                     
#SBATCH --job-name=es-demo-v2  # Name of the job
#SBATCH --output=/om/user/tiffany8/es-demo-v2.out
#SBATCH --error=/om/user/tiffany8/es-demo-v2.err
#SBATCH -n 4                     # 4 CPU core
#SBATCH --gres=gpu:a100:2            # 2 A100 GPU 
#SBATCH --mem=64G
#SBATCH --time=48:00:00               # Time limit (HH:MM:SS)
#SBATCH --partition=cbmm

set -x

whoami

source /om/user/tiffany8/miniconda/etc/profile.d/conda.sh
conda activate zero 

echo "check gpu status before running job"
nvidia-smi

cd /om/user/tiffany8/grpo-urop/TinyZero

export MODEL_NAME=/om/user/tiffany8/grpo-urop/TinyZero/model/Qwen2.5-3B-Instruct

python /om/user/tiffany8/grpo-urop/TinyZero/es_demo_v2.py \
  --model_name $MODEL_NAME \
  --hf_cache_dir /scratch/$USER/hf_cache \
  --mixed_precision 'bf16'
