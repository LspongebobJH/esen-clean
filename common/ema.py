import pytorch_lightning as pl

class EMACallback(pl.Callback):
    def __init__(self, decay=0.999, apply_ema_validation=True):
        super().__init__()
        self.decay = decay
        self.apply_ema_validation = apply_ema_validation
        self.shadow_params = {}
        self.backup_params = {}

    def on_train_start(self, trainer, pl_module):
        # 初始化影子参数（仅处理可训练参数）
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().detach()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 更新影子参数（仅处理可训练参数）
        for name, param in pl_module.named_parameters():
            if name in self.shadow_params:
                self.shadow_params[name] = self.decay * self.shadow_params[name] + (1 - self.decay) * param.detach()

    def on_validation_epoch_start(self, trainer, pl_module):
        if not self.apply_ema_validation:
            return
        
        # 备份原始参数并应用EMA参数
        self.backup_params = {
            name: param.data.clone().detach() 
            for name, param in pl_module.named_parameters()
        }
        for name, param in pl_module.named_parameters():
            if name in self.shadow_params:
                param.data.copy_(self.shadow_params[name])

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.apply_ema_validation:
            return
        
        # 恢复原始参数
        for name, param in pl_module.named_parameters():
            if name in self.backup_params:
                param.data.copy_(self.backup_params[name])
        self.backup_params.clear()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # 将EMA参数保存到checkpoint中（可选）
        checkpoint['ema_shadow_params'] = self.shadow_params

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # 从checkpoint加载EMA参数（可选）
        self.shadow_params = checkpoint.get('ema_shadow_params', {})
        print("[EMA] Loaded shadow params:", len(self.shadow_params))