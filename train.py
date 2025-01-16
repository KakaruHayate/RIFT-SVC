import os
import time

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader
import torch

from rift_svc import RF, DiT, LYNXNet
from rift_svc.dataset import collate_fn, load_svc_dataset
from rift_svc.lightning_module import RIFTSVCLightningModule


class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.step_start_time = None
        self.total_steps = None

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.start_time = time.time()
        self.total_steps = trainer.max_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        
        current_step = trainer.global_step
        total_steps = self.total_steps

        # Calculate elapsed time since training started
        elapsed_time = time.time() - self.start_time
        
        # Estimate average step time and remaining time
        average_step_time = elapsed_time / current_step if current_step > 0 else 0
        remaining_steps = total_steps - current_step
        remaining_time = average_step_time * remaining_steps if total_steps > 0 else 0

        # Format times with no leading zeros for hours
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours}:{minutes:02d}:{seconds:02d}"

        elapsed_time_str = format_time(elapsed_time)
        remaining_time_str = format_time(remaining_time)

        # Update the progress bar with loss, elapsed time, remaining time, and remaining steps
        self.train_progress_bar.set_postfix({
            "loss": f"{outputs['loss'].item():.4f}",
            "elapsed_time": elapsed_time_str + "/" + remaining_time_str,
            "remaining_steps": str(remaining_steps) + "/" + str(total_steps)
        })


def configure_optimizers(model, lr, betas, weight_decay, warmup_steps):
    from collections import defaultdict
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    specp_decay_params = defaultdict(list)
    specp_decay_lr = {}
    decay_params = []
    nodecay_params = []
    for n, p in param_dict.items():
        if p.dim() >= 2:
            if n.endswith('out.weight') or n.endswith('proj.weight'):
                fan_out, fan_in = p.shape[-2:]
                fan_ratio = fan_out / fan_in
                specp_decay_params[f"specp_decay_{fan_ratio:.2f}"].append(p)
                specp_decay_lr[f"specp_decay_{fan_ratio:.2f}"] = lr * fan_ratio
            else:
                decay_params.append(p)
        else:
            nodecay_params.append(p)
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay, 'lr': lr},
        {'params': nodecay_params, 'weight_decay': 0.0, 'lr': lr}
    ] + [
        {'params': params, 'weight_decay': weight_decay, 'lr': specp_decay_lr[group_name]}
        for group_name, params in specp_decay_params.items()
    ]
    
    optimizer = AdamWScheduleFree(optim_groups, betas=betas, warmup_steps=warmup_steps)
    return optimizer


def load_state_dict(model, state_dict, strict=False):
    """Load state dict while handling 'model.' prefix"""
    if any(k.startswith('model.') for k in state_dict.keys()):
        # Remove 'model.' prefix
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    return model.load_state_dict(state_dict, strict=strict)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    train_dataset = load_svc_dataset(
        data_dir=cfg.dataset.data_dir,
        meta_info_path=cfg.dataset.meta_info_path,
        max_frame_len=cfg.dataset.max_frame_len,
    )
    
    val_dataset = load_svc_dataset(
        data_dir=cfg.dataset.data_dir,
        meta_info_path=cfg.dataset.meta_info_path,
        max_frame_len=cfg.dataset.max_frame_len,
        split="test"
    )
    
    if 'lynxnet' in cfg.model.name:
        transformer = LYNXNet(
            **cfg.model.cfg,
            num_speaker=train_dataset.num_speakers,
            mel_channels=cfg.dataset.n_mel_channels,
        )
    else:
        transformer = DiT(
            **cfg.model.cfg,
            num_speaker=train_dataset.num_speakers,
            mel_dim=cfg.dataset.n_mel_channels,
        )

    rf = RF(
        transformer=transformer,
        num_mel_channels=cfg.dataset.n_mel_channels,
        whisper_drop_prob=cfg.model.get('whisper_drop_prob', 0.2),
        lognorm=cfg.model.get('lognorm', True),
    )

    # Load pretrained weights if specified
    if cfg.model.get('pretrained_path', None) is not None:
        state_dict = torch.load(cfg.model.pretrained_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Load only model weights, allowing mismatched keys for speaker embeddings
        missing_keys, unexpected_keys = load_state_dict(rf, state_dict)
        print(f"Loaded pretrained model from {cfg.model.pretrained_path}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    warmup_steps = int(cfg.training.max_steps * cfg.training.warmup_ratio)
    optimizer = configure_optimizers(
        rf, cfg.training.learning_rate, eval(cfg.training.betas), cfg.training.weight_decay, warmup_steps)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict['spk2idx'] = train_dataset.spk2idx
    model = RIFTSVCLightningModule(
        model=rf,
        optimizer=optimizer,
        cfg=cfg_dict
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('ckpts', cfg.training.wandb_run_name),
        filename='model-{step}',
        save_top_k=-1,
        save_last='link',
        every_n_train_steps=cfg.training.save_per_steps,
        save_weights_only=cfg.training.save_weights_only,
    )

    wandb_logger = WandbLogger(
        project=cfg.training.wandb_project,
        name=cfg.training.wandb_run_name,
        id=cfg.training.get('wandb_resume_id', None),
        resume='allow',
    )
    if wandb_logger.experiment.config:
        # Merge with existing config, giving priority to existing values
        wandb_logger.experiment.config.update(cfg_dict, allow_val_change=True)
    else:
        # If no existing config, set it directly
        wandb_logger.experiment.config.update(cfg_dict)


    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        accelerator='gpu',
        devices='auto',
        strategy='auto',
        precision='bf16-mixed',
        accumulate_grad_batches=cfg.training.grad_accumulation_steps,
        callbacks=[checkpoint_callback, CustomProgressBar()],
        logger=wandb_logger,
        val_check_interval=cfg.training.test_per_steps,
        check_val_every_n_epoch=None,
        gradient_clip_val=cfg.training.max_grad_norm,
        gradient_clip_algorithm='norm',
        log_every_n_steps=1,
    )

    optimizer.train()
    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size_per_gpu,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size_per_gpu,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_fn,
        ),
        ckpt_path=cfg.training.get('resume_from_checkpoint', None),
    )

if __name__ == "__main__":
    main()