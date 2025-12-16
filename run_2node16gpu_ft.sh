#!/usr/bin/env bash
set -ex
export USER_NAME="lijiahang"
export ENV_NAME="esen"
export PROJECT_NAME="esen"
export JOB_DIR="/mnt/shared-storage-user/${USER_NAME}/jobs/esen-clean/"

# --------------- Paths & Conda ---------------
cd ${JOB_DIR}
export PATH="/mnt/shared-storage-user/${USER_NAME}/miniconda3/bin:$PATH"
. /mnt/shared-storage-user/${USER_NAME}/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# --------------- W&B: offline to local disk ---------------
export WANDB_MODE=offline
export WANDB_PROJECT=${PROJECT_NAME}
export WANDB_DIR="/mnt/shared-storage-user/${USER_NAME}/wandb/${JOB_ID:-local_job}/node${NODE_RANK:-0}"
export WANDB_RUN_GROUP="${JOB_ID:-mp_group}"

NNODES="${NODE_COUNT:-2}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${PROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"

identifier="${PROJECT_NAME}_${ts}"

# --------------- model argument setting ---------------
# conservative
sampler_type="max_atoms"
# sampler_type="balanced" # after comparison, same as e2former new_balanced, thus use esen's balanced
precision=32 # not work when max_atoms
batch_size=32 # not work when max_atoms
regressor="esen_conservative"

ts="$(date +%Y%m%d_%H%M%S)"
name="esen_${ts}"
torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="${JOB_ID:-mptrj}" \
    run.py \
    trainer.num_nodes=2 \
    trainer.precision=${precision} \
    trainer.max_epochs=100 \
    lightning_module/regressor=${regressor} \
    lightning_module.scheduler_partial.scheduler.first_cycle_steps=147840 \
    lightning_module.load_energy_head=True \
    data_module.num_workers.train=8 \
    data_module.num_workers.val=8 \
    data_module.batch_size.train=${batch_size} \
    data_module.batch_size.val=${batch_size} \
    data_module.sampler_type=${sampler_type} \
    trainer.logger.name=${name} \
    trainer.logger.save_dir=/mnt/shared-storage-user/lijiahang/wandb/esen-clean/${name} \
    lightning_module.pretrained_ckpt='/mnt/shared-storage-user/lijiahang/wandb/esen-clean/esen_20251212_145457/esen/s8jkoo36/checkpoints/epoch\=59-step\=295680.ckpt'