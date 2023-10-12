#!/bin/bash -x
#SBATCH --account=cstdl
#SBATCH --nodes=160
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --output=/p/home/jusers/nakamura2/juwels/nakamura2/fmengine/%j_%N_log.out  # change this line to your output file

# Network Configuration
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10
export NCCL_ASYNC_ERROR_HANDLING=1

echo $SLURM_JOB_GPUS
echo $SLURM_NTASKS
echo $SLURM_NODELIST

# Convert SLURM_JOB_GPUS to an array
IFS=',' read -ra GPU_ARRAY <<< "$SLURM_JOB_GPUS"

# Get the number of GPUs from the length of the array
NUM_GPUS=${#GPU_ARRAY[@]}

export TOTAL_GPUS=$(($NUM_GPUS * $SLURM_NTASKS))
echo $TOTAL_GPUS

# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
master_addr="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)i"

export MASTER_ADDR=$master_addr

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
# export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Print System Information
echo "GPUs available to job: $SLURM_JOB_GPUS"
echo "Total tasks: $SLURM_NTASKS"

# Loop over all nodes
for ((i=0; i<$COUNT_NODE; i++))
do
    srun --nodes=1 --ntasks=1 -w "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n "$((i+1))p")" singularity run --nv \
    --home /p/home/jusers/nakamura2/juwels/:/home/nakamura2 \
    --bind /p/scratch/ccstdl/transformers_cache:/.hf_cache \
    --env HF_HOME=/.hf_cache \
    --env PYTHONPATH=/workspace \
    --bind /p/scratch/ccstdl/transformers_cache/pretrained_weights:/pretrained \
    --bind /p/scratch/ccstdl/transformers_cache/datasets:/datasets \
    --bind $PWD:/workspace \
    --pwd /workspace \
    fmsys_0.0.4.sif \
    torchrun \
    --master_addr "$MASTER_ADDR" \
    --master_port 12802 \
    --node_rank $i \
    --nnodes $SLURM_NTASKS \
    --nproc-per-node=$NUM_GPUS \
        cli/train.py \
        --output_dir /workspace/.cache/models \
        --init_ckpt /pretrained/llama-2-70b-hf-4shard \
        --data_path /datasets/prompt.jsonl \
        --max_seq_len 4096 \
        --train_steps 10 \
        --eval_steps 50 \
        --save_steps 1000 \
        --log_steps 1 \
        --pipe_parallel_size 1 \
        --model_parallel_size 4 \
        --use_flash_attn true \
        --use_fused_ops true \
        --deepspeed_config ./configs/llama.json &
done

wait
