#!/usr/bin/env python3
"""
Training script for QFinder model using PyTorch Lightning.

Usage:
    python training/scripts/train_QFinder.py --config configs/QFinder_config.yaml
    python training/scripts/train_QFinder.py --config configs/QFinder_config.yaml --resume_from_checkpoint path/to/checkpoint.ckpt
"""

import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import numpy as np
import torch
from collections import Counter
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.modules import QFinderLightningModule
from data import QFinderDataset

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

def load_config(config_path):
    """Load configuration from YAML file."""

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(config, group_name= "train"):
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
    if not h5_paths:
        raise ValueError(f"No HDF5 paths specified for {group_name} set")
    
    dataset = QFinderDataset(h5_paths=h5_paths, group_name=group_name)
    
    is_training = (group_name == "train")
 
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=is_training,
        num_workers=0,  # Must be 0 for HDF5 files
        pin_memory=config['training']['pin_memory']
    )
    
    return dataloader


def main():
    import time
    start_total = time.perf_counter()
    parser = argparse.ArgumentParser(description="Train QFinder model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    # Fast run to test the code
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run a quick test (1 batch, 1 epoch)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create data loaders
    print("Creating data loaders")
    train_loader = create_data_loaders(config, "train")
    val_loader = create_data_loaders(config, "val")
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("Computing class weights from training set")
    class_weights = compute_class_weights(
        train_loader.dataset,
        config['model']['num_classes']
    )

    print("Class weights:", class_weights.tolist())
    # Create model
    print("Creating model")
    model = QFinderLightningModule(
        num_classes=config['model']['num_classes'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_patience=config['lr_scheduler']['patience'],
        lr_scheduler_threshold=config['lr_scheduler']['threshold'],
        lr_scheduler_factor=config['lr_scheduler']['factor'],
        lr_scheduler_mode=config['lr_scheduler']['mode'],
	    class_weights=class_weights
    )
    
    callbacks = []
    
    # Checkpointing callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config['logging']['log_dir']) / config['logging']['name'] / "checkpoints",
        filename=config['checkpoint']['filename'],
        monitor=config['checkpoint']['monitor'],
        mode=config['checkpoint']['mode'],
        save_top_k=config['checkpoint']['save_top_k'],
        save_last=config['checkpoint']['save_last'],
        every_n_epochs=config['checkpoint']['every_n_epochs'],
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor=config['early_stopping']['monitor'],
        mode=config['early_stopping']['mode'],
        patience=config['early_stopping']['patience'],
        min_delta=config['early_stopping']['min_delta'],
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name=config['logging']['name']
    )
    
    # Create trainer
    trainer_config = config['trainer']
    trainer = pl.Trainer(
        accelerator=trainer_config['accelerator'],
        devices=trainer_config['devices'],
        precision=trainer_config['precision'],
        max_epochs=trainer_config['max_epochs'],
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=trainer_config.get('gradient_clip_val'),
        check_val_every_n_epoch=trainer_config['check_val_every_n_epoch'],
        log_every_n_steps=trainer_config['log_every_n_steps'],
        fast_dev_run=args.fast_dev_run
    )
    
    # Train
    print("Starting training")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint
    )
    
    end_total = time.perf_counter()
    total_time = end_total - start_total

    print(f"Total execution time: {total_time:.2f} seconds")

    print("Training completed")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
