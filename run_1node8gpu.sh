#!/usr/bin/env bash
set -ex
# --------------- W&B: offline to local disk ---------------
export USER_NAME=$(whoami)
export WANDB_MODE=offline
export WANDB_RUN_GROUP="${JOB_ID:-mp_group}"
ts="$(date +%Y%m%d_%H%M%S)"
NNODES="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${PROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"

# data_module.sampler_type: conservative: max_atoms; direct-force: balanced
# trainer.precision: conservative training: 32; direct-force pretraining: 16-mixed
# data_module.batch_size: conservative training: sampler adaptive batch-size, specifying it not work; direct-force training: 300

if [ "$1" == "conservative" ]; then
    sampler_type="max_atoms"
    precision=32 # not work
    batch_size=32 # not work
    regressor="esen_conservative"
elif [ "$1" == "direct-force" ]; then
    sampler_type="balanced"
    precision=16-mixed
    # batch_size=300 # OOM
    batch_size=32
    regressor="esen_direct"
else
    echo "Please specify training type: conservative or direct-force"
    exit 1
fi

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
    trainer.num_nodes=1 \
    trainer.precision=${precision} \
    trainer.max_epochs=60 \
    lightning_module/regressor=${regressor} \
    lightning_module.scheduler_partial.scheduler.first_cycle_steps=147840 \
    data_module.num_workers.train=8 \
    data_module.num_workers.val=8 \
    data_module.batch_size.train=${batch_size} \
    data_module.batch_size.val=${batch_size} \
    data_module.sampler_type=${sampler_type} \
    trainer.logger.name=${name} \
    trainer.logger.save_dir=/mnt/shared-storage-user/lijiahang/wandb/esen-clean/${name}