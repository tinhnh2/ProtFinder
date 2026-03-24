#!/usr/bin/env python3
"""
Run JOINT TRAINING + FINE TUNING in ONE execution
"""

import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.modules import QFinderLightningModule
from data import QFinderDataset


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_loader(h5_path, group, batch, pin):
    ds = QFinderDataset([h5_path], group_name=group)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=(group == "train"),
        num_workers=0,
        pin_memory=pin,
    )


def ckpt_callback(save_dir, prefix):
    return ModelCheckpoint(
        dirpath=save_dir,
        filename=f"QFinder-{prefix}-{{epoch}}-{{val_acc:.4f}}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False,
    )


def create_data_loaders(config, group_name="train", action="joint", batch_size=64):
    """
    Create data loaders.

    Args:
        config: Configuration dictionary
        group_name: "train", "val", or "test"

    Returns:
        DataLoader instance
    """

    # Use train_val_h5_paths for both train and val groups
    h5_paths = config['data']['train_val_h5_paths']
    if action=="joint":
        h5_paths = config['data']['joint_h5_paths']
    if action=="tuning":
        h5_paths = config['data']['tuning_h5_paths']
    if not h5_paths:
        raise ValueError(f"No HDF5 paths specified for {group_name} set")

    dataset = QFinderDataset(h5_paths=h5_paths, group_name=group_name)

    is_training = (group_name == "train")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=0,  # Must be 0 for HDF5 files
        pin_memory=config['training']['pin_memory']
    )

    return dataloader

def freeze_backbone(model):
    """
    Freeze all convolution + SE layers, keep FC trainable
    """
    for name, param in model.model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False

# ------------------------------------------------------------
# FREEZE / THAW for CNN
# ------------------------------------------------------------
def freeze_all_conv(model):
    """
    Freeze all convolutional layers, keep classifier trainable
    """
    for name, module in model.model.named_modules():
        if "conv" in name or "block" in name:
            for p in module.parameters():
                p.requires_grad = False

def thaw_top_conv_blocks(model, n_blocks=1):
    """
    Unfreeze top convolutional blocks
    Assumes blocks are named: conv_block1, conv_block2, ...
    """
    blocks = [
        (name, m) for name, m in model.model.named_modules()
        if "block" in name
    ]
    blocks = sorted(blocks, key=lambda x: x[0])
    for name, block in blocks[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained_ckpt", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_dir = Path(cfg['logging']['log_dir']) / cfg['logging']['name'] / "checkpoints"

    # =========================================================
    # LOAD BASE MODEL (SIMULATION PRETRAINED)
    # =========================================================
    model = QFinderLightningModule.load_from_checkpoint(
        args.pretrained_ckpt,
        strict=False,
        num_classes=cfg["model"]["num_classes"],
        learning_rate=cfg["training"]["learning_rate_tuning"],
        weight_decay=cfg["training"]["weight_decay"],
        lr_scheduler_patience=cfg["lr_scheduler"]["patience"],
        lr_scheduler_threshold=cfg["lr_scheduler"]["threshold"],
        lr_scheduler_factor=cfg["lr_scheduler"]["factor"],
        lr_scheduler_mode=cfg["lr_scheduler"]["mode"],
    )

    # =========================================================
    # PHASE 1: JOINT TRAINING (SIM + REAL)
    # =========================================================
    print("\n==============================")
    print(" PHASE 1: JOINT TRAINING")
    print("==============================")

    train_loader = create_data_loaders(cfg, "train","joint",64)
    val_loader = create_data_loaders(cfg, "val","joint", 64)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    joint_logger = TensorBoardLogger(
        save_dir=cfg["logging"]["log_dir"],
        name=cfg["logging"]["name"] + "_joint",
    )

    joint_ckpt = ckpt_callback(ckpt_dir, "joint")

    trainer = pl.Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        precision=cfg["trainer"]["precision"],
        max_epochs=20,
        logger=joint_logger,
        callbacks=[joint_ckpt],
        log_every_n_steps=20,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    joint_last_ckpt = joint_ckpt.last_model_path
    print(f"Joint-training done. Last checkpoint: {joint_last_ckpt}")

    # =========================================================
    # LOAD JOINT MODEL FOR FINETUNING
    # =========================================================
    model = QFinderLightningModule.load_from_checkpoint(
        joint_last_ckpt,
        strict=False,
    )

    # =========================================================
    # PHASE 2: FINE TUNING (REAL ONLY – FREEZE → THAW)
    # =========================================================
    print("\n==============================")
    print(" PHASE 2: FINE TUNING")
    print("==============================")

    train_loader = create_data_loaders(cfg, "train","joint",32)
    val_loader = create_data_loaders(cfg, "val","joint", 32)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    finetune_logger = TensorBoardLogger(
        save_dir=cfg["logging"]["log_dir"],
        name=cfg["logging"]["name"] + "_finetune",
    )

    """
    # -------------------------
    # Stage 2.1: Freeze conv
    # -------------------------
    print("→ Freeze all conv blocks")
    freeze_all_conv(model)

    freeze_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="finetune_freeze-{epoch}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        precision=cfg["trainer"]["precision"],
        max_epochs=5,
        logger=finetune_logger,
        callbacks=[freeze_ckpt],
    )
    trainer.fit(model, train_loader, val_loader)

    # -------------------------
    # Stage 2.2: Thaw top conv
    # -------------------------
    print("→ Thaw top conv block")
    thaw_top_conv_blocks(model, n_blocks=1)

    thaw_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="finetune_thaw-{epoch}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        precision=cfg["trainer"]["precision"],
        max_epochs=10,
        logger=finetune_logger,
        callbacks=[thaw_ckpt],
    )
    """
    # ---------- Stage 2.1: Freeze backbone ----------
    print("→ Stage 2.1: Freeze backbone")
    freeze_backbone(model)
    thaw_top_conv_blocks(model, n_blocks=1)

    freeze_ckpt = ckpt_callback(ckpt_dir, "finetune")

    trainer = pl.Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        precision=cfg["trainer"]["precision"],
        max_epochs=10,
        logger=finetune_logger,
        callbacks=[freeze_ckpt],
    )
    
    trainer.fit(model, train_loader, val_loader)

    print("\n==============================")
    print(" TRAINING PIPELINE FINISHED ")
    print("==============================")
    print("Best final model:", freeze_ckpt.best_model_path)


if __name__ == "__main__":
    main()

