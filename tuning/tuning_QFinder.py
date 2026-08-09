#!/usr/bin/env python3
"""
Run JOINT TRAINING + FINE TUNING in ONE execution
"""

import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from collections import Counter
import torch
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
        h5_paths = config['data']['real_h5_paths']
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

# ------------------------------------------------------------
# FREEZE / THAW for CNN
# ------------------------------------------------------------
# QFinderModel's actual submodule names (see QFinder.py) are
# conv1, se1, conv2, se2, conv3, se3, conv4, se4, avgpool, fc
# — NOT "conv_block1", "conv_block2", ... The previous version of
# these functions filtered on the substring "block", which never
# matched anything, so thaw_top_conv_blocks() was a silent no-op.
#
# Block 1 (conv1+se1) is closest to the input (most generic features).
# Block 4 (conv4+se4) is closest to the classifier (most task/domain
# specific) — this is the one we want to thaw first for fine-tuning.
_N_BLOCKS = 4


def _block_modules(model, block_id: int):
    """Return the (conv_i, se_i) submodule pair for a given block id (1-4)."""
    conv_mod = getattr(model.model, f"conv{block_id}")
    se_mod = getattr(model.model, f"se{block_id}")
    return conv_mod, se_mod


def freeze_backbone(model):
    """
    Freeze all conv+SE blocks, keep FC trainable.

    Freezing here means two things, both necessary:
    1. requires_grad=False on every parameter, so no gradient updates
       the weights.
    2. .eval() on every frozen submodule, so BatchNorm layers STOP
       updating running_mean/running_var from the fine-tuning data.
       requires_grad alone does NOT stop this: running stats are
       buffers, not parameters, and get updated on every forward pass
       while the module is in train() mode.

    The frozen submodules are registered via model.set_frozen_modules()
    so that PyTorch Lightning's automatic model.train() calls (once per
    epoch) don't silently put them back into train mode.
    """
    frozen_modules = []
    for block_id in range(1, _N_BLOCKS + 1):
        conv_mod, se_mod = _block_modules(model, block_id)
        for p in conv_mod.parameters():
            p.requires_grad = False
        for p in se_mod.parameters():
            p.requires_grad = False
        conv_mod.eval()
        se_mod.eval()
        frozen_modules.extend([conv_mod, se_mod])

    model.set_frozen_modules(frozen_modules)


def thaw_top_conv_blocks(model, n_blocks=1):
    """
    Unfreeze the n_blocks conv+SE pairs closest to the classifier
    (block4 first, then block3, ...). Must be called AFTER
    freeze_backbone(model).
    """
    if n_blocks < 1 or n_blocks > _N_BLOCKS:
        raise ValueError(f"n_blocks must be between 1 and {_N_BLOCKS}, got {n_blocks}")

    top_block_ids = range(_N_BLOCKS, _N_BLOCKS - n_blocks, -1)  # e.g. n_blocks=1 -> [4]

    still_frozen = list(model._frozen_modules)
    for block_id in top_block_ids:
        conv_mod, se_mod = _block_modules(model, block_id)
        for p in conv_mod.parameters():
            p.requires_grad = True
        for p in se_mod.parameters():
            p.requires_grad = True
        conv_mod.train()
        se_mod.train()
        # No longer frozen -> remove from the "pin to eval()" list so
        # future automatic .train() calls keep training its BatchNorm.
        still_frozen = [m for m in still_frozen if m is not conv_mod and m is not se_mod]

    model.set_frozen_modules(still_frozen)

def compute_class_weights(dataset, num_classes):
    """
    Compute balanced class weights from training dataset.
    """
    labels = []

    for i in range(len(dataset)):
        _, y, key = dataset[i]
        labels.append(int(y))

    counts = Counter(labels)
    total = sum(counts.values())

    weights = []
    for c in range(num_classes):
        wc = total / (num_classes * counts.get(c, 1))
        weights.append(wc)

    weights = torch.tensor(weights, dtype=torch.float32)
    return weights

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained_ckpt", required=True)
    parser.add_argument("--class_weights", action="store_true",
                        help="Enable class weighting (use uniform loss)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_dir = Path(cfg['logging']['log_dir']) / cfg['logging']['name'] / "checkpoints"

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
    print("Computing class weights from training set")
    class_weights = None
    if args.class_weights:
        class_weights = compute_class_weights(
            train_loader.dataset,
            cfg['model']['num_classes']
        )
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
        class_weights=class_weights
    )
    joint_logger = TensorBoardLogger(
        save_dir=cfg["logging"]["log_dir"],
        name=cfg["logging"]["name"] + "_joint",
    )

    joint_ckpt = ckpt_callback(ckpt_dir, "joint")
    callbacks = []
    callbacks.append(joint_ckpt)
    early_stopping = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=cfg["early_stopping"]["patience"],
        min_delta=cfg["early_stopping"]["min_delta"],
        verbose=True
    )
    callbacks.append(early_stopping)

    trainer = pl.Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        precision=cfg["trainer"]["precision"],
        max_epochs=cfg["trainer"]["max_epochs_joint"],
        logger=joint_logger,
        callbacks=callbacks,
        log_every_n_steps=20,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    joint_last_ckpt = joint_ckpt.best_model_path
    print(f"Joint-training done. Last checkpoint: {joint_last_ckpt}")


    # =========================================================
    # PHASE 2: FINE TUNING (REAL ONLY – FREEZE → THAW)
    # =========================================================
    print("\n==============================")
    print(" PHASE 2: FINE TUNING")
    print("==============================")

    train_loader = create_data_loaders(cfg, "train","tuning",32)
    val_loader = create_data_loaders(cfg, "val","tuning", 32)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    if args.class_weights:
        class_weights = compute_class_weights(
            train_loader.dataset,
            cfg['model']['num_classes']
        )
	# =========================================================
    # LOAD JOINT MODEL FOR FINETUNING
    # =========================================================
    model = QFinderLightningModule.load_from_checkpoint(
        joint_last_ckpt,
        strict=False,
        class_weights=class_weights
    )
    finetune_logger = TensorBoardLogger(
        save_dir=cfg["logging"]["log_dir"],
        name=cfg["logging"]["name"] + "_finetune",
    )

    # ---------- Stage 2.1: Freeze backbone ----------
    print("→ Stage 2.1: Freeze backbone")
    freeze_backbone(model)
    thaw_top_conv_blocks(model, n_blocks=1)

    freeze_ckpt = ckpt_callback(ckpt_dir, "finetune")

    callbacks = []
    callbacks.append(freeze_ckpt)
    early_stopping = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=cfg["early_stopping"]["patience"],
        min_delta=cfg["early_stopping"]["min_delta"],
        verbose=True
    )
    callbacks.append(early_stopping)
    trainer = pl.Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        precision=cfg["trainer"]["precision"],
        max_epochs=cfg["trainer"]["max_epochs_tuning"],
        logger=finetune_logger,
        callbacks=callbacks,
    )
    
    trainer.fit(model, train_loader, val_loader)

    print("\n==============================")
    print(" TRAINING PIPELINE FINISHED ")
    print("==============================")
    print("Best final model:", freeze_ckpt.best_model_path)


if __name__ == "__main__":
    main()
