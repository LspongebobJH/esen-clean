import os

import pytorch_lightning as pl

# from datetime import datetime



class CycleCheckpointCallback(pl.Callback):
    def __init__(self, cycle_steps: int, save_dir: str = None):
        super().__init__()
        self.checkpoint_interval = cycle_steps

        if save_dir is None:
            self.save_dir = f"./cycle_checkpoints"
        else:
            self.save_dir = save_dir

    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            os.makedirs(self.save_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.is_global_zero:
            current_step = trainer.global_step
            if (current_step + 1) % self.checkpoint_interval == 0:
                checkpoint_name = f"step={current_step}.ckpt"
                checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

                print(f"Saving cycle checkpoint at step {current_step}: {checkpoint_path}")
                trainer.save_checkpoint(checkpoint_path)
                print(f"Saved cycle checkpoint at step {current_step}: {checkpoint_path}")
