#!/bin/bash -x
#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=develbooster
#SBATCH --output=/p/home/jusers/nakamura2/juwels/nakamura2/fmengine/%j_0_log.out  # change this line to your output file


srun singularity run --nv \
--home /p/home/jusers/nakamura2/juwels/:/home/nakamura2 \
--bind /p/scratch/ccstdl/transformers_cache:/.hf_cache \
--env HF_HOME=/.hf_cache \
--env PYTHONPATH=/workspace \
--bind /p/scratch/ccstdl/transformers_cache/pretrained_weights:/pretrained \
--bind /p/scratch/ccstdl/transformers_cache/datasets:/datasets \
--bind $PWD:/workspace \
--pwd /workspace \
fmsys_0.0.4.sif \
bash finetune_llama.sh
# bash hf_model_to_fm.sh