import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import RASFinderModel


class RASFinderLightningModule(pl.LightningModule):

    def __init__(
		self,
        input_dim: int,
        summary_dim: int,
        num_classes: int,
        num_heads: int,
        num_layers: int,
        dim_model: int,
        dim_feedforward: int,
        learning_rate: float,
        weight_decay: float,
        lr_scheduler_patience: int,
        lr_scheduler_threshold: float,
        lr_scheduler_factor: float,
        lr_scheduler_mode: str,
        class_weights: torch.Tensor = None  # NEW: optional class weight tensor
	):
        super().__init__()
        if class_weights is not None:
            self.save_hyperparameters(ignore=['class_weights'])
        else:
            self.save_hyperparameters()
        torch.set_float32_matmul_precision("high")
        self.model = RASFinderModel(
            input_dim=input_dim,
            summary_dim=summary_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_model=dim_model,
            dim_feedforward=dim_feedforward
        )

        # Loss: weighted CrossEntropy if class_weights provided
        if class_weights is not None:
            print(f"[RASFinder] Using weighted CrossEntropyLoss: {class_weights.tolist()}")
            self.register_buffer('class_weights', class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            print("[RASFinder] Using uniform CrossEntropyLoss (no class weights)")
            self.register_buffer('class_weights', torch.ones(num_classes))
            self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, sitewise_feature, lengths, summary_feature):
        return self.model(sitewise_feature, lengths, summary_feature)

    def training_step(self, batch, batch_idx):

        sitewise_features, summary_features, lengths, labels, keys = batch

        logits = self(sitewise_features, lengths, summary_features)
        loss = self.criterion(logits, labels)

        self.train_accuracy(logits, labels)

        self.log_dict(
            {"train_loss": loss, "train_acc": self.train_accuracy},
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):

        sitewise_features, summary_features, lengths, labels, keys = batch

        logits = self(sitewise_features, lengths, summary_features)
        loss = self.criterion(logits, labels)

        self.val_accuracy(logits, labels)

        self.log_dict(
            {"val_loss": loss, "val_acc": self.val_accuracy},
            on_epoch=True,
            prog_bar=True
        )

        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:

        sitewise_features, summary_features, lengths, labels, keys = batch

        # Forward pass
        logits = self(sitewise_features, lengths, summary_features)
        loss = self.criterion(logits, labels)
        self.test_accuracy(logits, labels)

        # Logging
        values = {"test_loss": loss, "test_acc": self.test_accuracy}
        self.log_dict(values, on_step=False, on_epoch=True, logger=True)

        return {
            "loss":   loss,
            "logits": logits.detach(),
            "labels": labels.detach(),
            "keys": keys
        }

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=self.hparams.lr_scheduler_mode,
            factor=self.hparams.lr_scheduler_factor,
            patience=self.hparams.lr_scheduler_patience,
            threshold=self.hparams.lr_scheduler_threshold,
            threshold_mode='abs'
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc"
            }
        }

