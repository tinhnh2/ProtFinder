#!/usr/bin/env python3
"""
Fine-tuning RASFinder model on real data using pretrained model
trained on simulation data.
"""

import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

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
        batch_size=config["training"]["batch_size_tuning"],
        shuffle=(group_name == "train"),
        num_workers=0,
        pin_memory=config["training"]["pin_memory"],
        collate_fn=collate_fn_rasfinder
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
        strict=False
    )

    #if args.freeze_transformer:
        #print("Freezing transformer backbone...")
        #freeze_transformer_backbone(model)

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------
    callbacks = []

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
        accumulate_grad_batches=4
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
        strict=False
    )

    if args.freeze_transformer:
        print("Freezing transformer backbone...")
        freeze_transformer_backbone(model)

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------
    callbacks = []

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
        accumulate_grad_batches=4
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
