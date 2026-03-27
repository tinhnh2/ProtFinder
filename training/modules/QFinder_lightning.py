"""
PyTorch Lightning module for QFinder model training.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import QFinderModel


class QFinderLightningModule(pl.LightningModule):
    """
    Lightning module for training QFinder model.
    
    Handles:
    - Training and validation steps
    - Automatic mixed precision
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        weight_decay: float,
        lr_scheduler_patience: int,
        lr_scheduler_threshold: float,
        lr_scheduler_factor: float,
        lr_scheduler_mode: str,
        class_weights: torch.Tensor = None 
    ):
        """
        Args:
            num_classes: Number of substitution model classes
            learning_rate: Initial learning rate
            weight_decay: Weight decay for AdamW optimizer
            lr_scheduler_patience: Patience for learning rate scheduler
            lr_scheduler_threshold: Minimum change to qualify as improvement
            lr_scheduler_factor: Factor by which learning rate is reduced
            lr_scheduler_mode: 'max' for accuracy, 'min' for loss
        
        Note:
            All parameters should be provided from configuration file.
            When loading from checkpoint, hyperparameters are automatically restored.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = QFinderModel(num_classes=num_classes)
        
        # Loss function
        #self.criterion = nn.CrossEntropyLoss()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Learning rate scheduler settings
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_threshold = lr_scheduler_threshold
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_mode = lr_scheduler_mode
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Tuple of (features, labels)
            batch_idx: Batch index
        
        Returns:
            Loss value
        """
        features, labels, keys = batch
        
        # Forward pass
        logits = self(features)
        loss = self.criterion(logits, labels)
        self.train_accuracy(logits, labels)
        
        # Logging
        values = {"train_loss": loss, "train_acc": self.train_accuracy}
        self.log_dict(values, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:

        features, labels, keys = batch
        
        # Forward pass
        logits = self(features)
        loss = self.criterion(logits, labels)
        self.val_accuracy(logits, labels)
        
        # Logging
        values = {"val_loss": loss, "val_acc": self.val_accuracy}
        self.log_dict(values, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:

        features, labels, keys = batch
        
        # Forward pass
        logits = self(features)
        loss = self.criterion(logits, labels)
        self.test_accuracy(logits, labels)
        
        # Logging
        values = {"test_loss": loss, "test_acc": self.test_accuracy}
        self.log_dict(values, on_step=False, on_epoch=True, logger=True)
        
        return {
        "logits": logits.detach(),
        "labels": labels.detach(),
        "keys": keys
    }
    

    def configure_optimizers(self):
        """
        Set up optimizer and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and lr_scheduler
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=self.lr_scheduler_mode,
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            threshold=self.lr_scheduler_threshold,
            threshold_mode='abs',
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "interval": "epoch",
                "frequency": 1
            }
        }
