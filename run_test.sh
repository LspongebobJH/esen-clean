sampler_type="balanced" # after comparison, same as e2former new_balanced, thus use esen's balanced
precision=32 # not work when max_atoms
batch_size=32 # not work when max_atoms
regressor="esen_conservative"
ts="$(date +%Y%m%d_%H%M%S)"
name="esen_${ts}"

python run.py \
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