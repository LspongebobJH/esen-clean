#!/bin/bash
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline

torchrun --standalone --nproc_per_node=1 \
    --module run \
    trainer.max_epochs=1 \
    trainer.strategy.process_group_backend='gloo' \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.logger.name='demo' \
    data_module.batch_size.train=2 \
    data_module.batch_size.val=2 \
    lightning_module.use_denoising_pos=false \
    lightning_module.scheduler_partial.scheduler.first_cycle_steps=172 \
    lightning_module.scheduler_partial.scheduler.warmup_steps=17 