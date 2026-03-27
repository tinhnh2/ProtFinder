#!/usr/bin/env python3
"""
Fine-tuning RASFinder model on real data using pretrained model
trained on simulation data.
"""

import argparse
import numpy as np
import yaml
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from collections import Counter
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.modules import RASFinderLightningModule
from data import RASFinderDataset, collate_fn_rasfinder
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def make_collate_fn(max_seq_len=None):
    """
    Tạo collate_fn có hỗ trợ truncation sequence.
 
    Real MSA thường có số sites lớn hơn nhiều so với simulated MSA.
    Khi pad toàn bộ sequence trong batch lên cùng độ dài,
    tensor (B, max_n_sites, 23) sẽ bùng nổ bộ nhớ.
 
    max_seq_len: Nếu được đặt, truncate tất cả sequence xuống tối đa max_seq_len sites.
    """
    from typing import List, Tuple
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        sitewise_features, summary_features, labels, keys = zip(*batch)
 
        # Truncate nếu cần
        if max_seq_len is not None:
            sitewise_features = [
                f[:max_seq_len] if f.shape[0] > max_seq_len else f
                for f in sitewise_features
            ]
 
        lengths = torch.tensor([f.shape[0] for f in sitewise_features], dtype=torch.long)
        max_length = lengths.max().item()
 
        padded_sitewise = []
        for feature in sitewise_features:
            n_sites = feature.shape[0]
            if n_sites < max_length:
                padding = torch.zeros(max_length - n_sites, feature.shape[1], dtype=feature.dtype)
                padded = torch.cat([feature, padding], dim=0)
            else:
                padded = feature
            padded_sitewise.append(padded)
 
        sitewise_batch = torch.stack(padded_sitewise)   # (B, max_n_sites, 23)
        summary_batch  = torch.stack(list(summary_features))   # (B, 10)
        labels_batch   = torch.stack(list(labels))             # (B,)
 
        return sitewise_batch, summary_batch, lengths, labels_batch, list(keys)
 
    return collate_fn


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute balanced class weights from label array.

    Formula: weight[c] = total_samples / (num_classes * count[c])
    This is equivalent to sklearn's 'balanced' strategy.

    Args:
        y: Numpy array of integer labels
        num_classes: Total number of classes

    Returns:
        FloatTensor of shape (num_classes,)
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    total = len(y)
    weights = total / (num_classes * np.where(counts == 0, 1, counts))
    return torch.tensor(weights, dtype=torch.float32)


def collect_labels(dataloader) -> np.ndarray:
    """Collect all labels from a RASFinder DataLoader (batches: sitewise, summary, lengths, labels)."""
    all_labels = []
    for batch in dataloader:
        labels = batch[3]
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels)


"""
def compute_class_weights(dataset, num_classes):
    labels = []

    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(int(y))

    counts = Counter(labels)
    total = sum(counts.values())

    weights = []
    for c in range(num_classes):
        wc = total / (num_classes * counts.get(c, 1))
        weights.append(wc)

    weights = torch.tensor(weights, dtype=torch.float32)
    return weights
"""

def create_dataloader(config, group_name, data_type):
    h5_paths = config["data"]["joint_h5_paths"]
    if data_type == "tuning":
        h5_paths = config["data"]["real_h5_paths"]
    if not h5_paths:
        raise ValueError("No real HDF5 paths provided for RASFinder")

    dataset = RASFinderDataset(
        h5_paths=h5_paths,
        group_name=group_name
    )

    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(group_name == "train"),
        num_workers=0,
        pin_memory=config["training"]["pin_memory"],
        #collate_fn=collate_fn_rasfinder
		collate_fn=make_collate_fn(
            max_seq_len=config["trainer"]["max_seq_len"]
        )
    )


def freeze_transformer_backbone(model):
    """
    Freeze:
      - fc1 (projection)
      - transformer encoder
    Keep:
      - fc2, fc3 trainable
    """
    for name, param in model.model.named_parameters():
        if name.startswith("fc2") or name.startswith("fc3"):
            param.requires_grad = True
        else:
            param.requires_grad = False


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune RASFinder on real data"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to fine-tuning YAML config"
    )

    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        required=True,
        help="Path to pretrained RASFinder checkpoint (.ckpt)"
    )

    parser.add_argument(
        "--freeze_transformer",
        action="store_true",
        help="Freeze transformer encoder, train only classifier head"
    )

    parser.add_argument("--class_weights", 
        action="store_true",
        help="Enable class weighting (use uniform loss)")
    args = parser.parse_args()

    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------
    config = load_config(args.config)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    
    print("Loading RASFinder dataset for joint training...")
    train_loader = create_dataloader(config, "train", "joint")
    val_loader = create_dataloader(config, "val", "joint")

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    class_weights = None
    num_classes = 4
    if args.class_weights:
        print("\nComputing class weights from simulation + real training data...")
        train_labels = collect_labels(train_loader)
        classes, counts = np.unique(train_labels, return_counts=True)
        total = len(train_labels)

        print("Class Distribution:")
        class_names = ['None', '+I', '+G', '+G+I']
        for cls, cnt in zip(classes, counts):
            name = class_names[cls] if cls < len(class_names) else str(cls)
            print(f"  Class {cls} ({name}): {cnt} samples ({100*cnt/total:.1f}%)")

        class_weights = compute_class_weights(train_labels, num_classes)
        print(f"Class weights: {class_weights.tolist()}")

    # --------------------------------------------------------
    # Load pretrained model
    # --------------------------------------------------------
    print("Loading pretrained RASFinder model...")

    model = RASFinderLightningModule.load_from_checkpoint(
        checkpoint_path=args.pretrained_ckpt,
        input_dim=config["model"]["input_dim"],
        summary_dim=config["model"]["summary_dim"],
        num_classes=config["model"]["num_classes"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dim_model=config["model"]["dim_model"],
        dim_feedforward=config["model"]["dim_feedforward"],
        learning_rate=config["training"]["learning_rate_joint"],
        weight_decay=config["training"]["weight_decay"],
        lr_scheduler_patience=config["lr_scheduler"]["patience"],
        lr_scheduler_threshold=config["lr_scheduler"]["threshold"],
        lr_scheduler_factor=config["lr_scheduler"]["factor"],
        lr_scheduler_mode=config["lr_scheduler"]["mode"],
        strict=False,
        class_weights=class_weights
    )

    #if args.freeze_transformer:
        #print("Freezing transformer backbone...")
        #freeze_transformer_backbone(model)

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------
    callbacks = []
    if hasattr(model.model, 'use_checkpoint'):
        model.model.use_checkpoint = True

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config["logging"]["log_dir"])
        / config["logging"]["name"]
        / "checkpoints",
        filename="RASFinder-joint-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    early_stopping = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=config["early_stopping"]["patience"],
        min_delta=config["early_stopping"]["min_delta"],
        verbose=True
    )
    callbacks.append(early_stopping)

    # --------------------------------------------------------
    # Logger
    # --------------------------------------------------------
    logger = TensorBoardLogger(
        save_dir=config["logging"]["log_dir"],
        name=config["logging"]["name"] + "_joint"
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------
    trainer_cfg = config["trainer"]
    trainer = pl.Trainer(
        accelerator=trainer_cfg["accelerator"],
        devices=trainer_cfg["devices"],
        precision=trainer_cfg["precision"],
        max_epochs=trainer_cfg["max_epochs_joint"],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=trainer_cfg["log_every_n_steps"],
        accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
		gradient_clip_val=config["training"]["gradient_clip_val"]
    )

    # --------------------------------------------------------
    # Fine-tune
    # --------------------------------------------------------
    print("Starting RASFinder joint training..")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    print("Joint training completed.")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    
    #++++++++++++++++++++++++++++++++++++++++++#
    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    print("Loading real RASFinder dataset for tuning...")
    train_loader = create_dataloader(config, "train","tuning")
    val_loader = create_dataloader(config, "val", "tuning")

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    class_weights = None
    if args.class_weights:
        print("\nComputing class weights from real training data...")
        train_labels = collect_labels(train_loader)
        classes, counts = np.unique(train_labels, return_counts=True)
        total = len(train_labels)

        print("Class Distribution:")
        class_names = ['None', '+I', '+G', '+G+I']
        for cls, cnt in zip(classes, counts):
            name = class_names[cls] if cls < len(class_names) else str(cls)
            print(f"  Class {cls} ({name}): {cnt} samples ({100*cnt/total:.1f}%)")

        class_weights = compute_class_weights(train_labels, num_classes)
        print(f"Class weights: {class_weights.tolist()}")

    # --------------------------------------------------------
    # Load pretrained model
    # --------------------------------------------------------
    print("Loading joint pretrained RASFinder model...")

    model = RASFinderLightningModule.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path,
        input_dim=config["model"]["input_dim"],
        summary_dim=config["model"]["summary_dim"],
        num_classes=config["model"]["num_classes"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dim_model=config["model"]["dim_model"],
        dim_feedforward=config["model"]["dim_feedforward"],
        learning_rate=config["training"]["learning_rate_tuning"],
        weight_decay=config["training"]["weight_decay"],
        lr_scheduler_patience=config["lr_scheduler"]["patience"],
        lr_scheduler_threshold=config["lr_scheduler"]["threshold"],
        lr_scheduler_factor=config["lr_scheduler"]["factor"],
        lr_scheduler_mode=config["lr_scheduler"]["mode"],
        strict=False,
        class_weights=class_weights
    )

    if args.freeze_transformer:
        print("Freezing transformer backbone...")
        freeze_transformer_backbone(model)

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------
    callbacks = []
    if hasattr(model.model, 'use_checkpoint'):
        model.model.use_checkpoint = True

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config["logging"]["log_dir"])
        / config["logging"]["name"]
        / "checkpoints",
        filename="RASFinder-finetune-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    early_stopping = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=config["early_stopping"]["patience"],
        min_delta=config["early_stopping"]["min_delta"],
        verbose=True
    )
    callbacks.append(early_stopping)

    # --------------------------------------------------------
    # Logger
    # --------------------------------------------------------
    logger = TensorBoardLogger(
        save_dir=config["logging"]["log_dir"],
        name=config["logging"]["name"] + "_finetune"
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------
    trainer_cfg = config["trainer"]
    trainer = pl.Trainer(
        accelerator=trainer_cfg["accelerator"],
        devices=trainer_cfg["devices"],
        precision=trainer_cfg["precision"],
        max_epochs=trainer_cfg["max_epochs_tuning"],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=trainer_cfg["log_every_n_steps"],
        accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
        gradient_clip_val=config["training"]["gradient_clip_val"]
    )

    # --------------------------------------------------------
    # Fine-tune
    # --------------------------------------------------------
    print("Starting RASFinder fine-tuning...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    print("Fine-tuning completed.")
    print(f"Best model: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
