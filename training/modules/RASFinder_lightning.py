import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import RASFinderModel


class RASFinderLightningModule(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        torch.set_float32_matmul_precision("high")

        self.model = RASFinderModel(
            input_dim=kwargs["input_dim"],
            summary_dim=kwargs["summary_dim"],
            num_classes=kwargs["num_classes"],
            num_heads=kwargs["num_heads"],
            num_layers=kwargs["num_layers"],
            dim_model=kwargs["dim_model"],
            dim_feedforward=kwargs["dim_feedforward"],
        )


        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=kwargs["num_classes"])
        self.val_accuracy = Accuracy(task="multiclass", num_classes=kwargs["num_classes"])
        self.test_accuracy = Accuracy(task="multiclass", num_classes=kwargs["num_classes"])

    def forward(self, sitewise_feature, lengths, summary_feature):
        return self.model(sitewise_feature, lengths, summary_feature)

    def training_step(self, batch, batch_idx):

        sitewise_features, summary_features, lengths, labels = batch

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

        sitewise_features, summary_features, lengths, labels = batch

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

        sitewise_features, summary_features, lengths, labels = batch

        # Forward pass
        logits = self(sitewise_features, lengths, summary_features)
        loss = self.criterion(logits, labels)
        self.test_accuracy(logits, labels)

        # Logging
        values = {"test_loss": loss, "test_acc": self.test_accuracy}
        self.log_dict(values, on_step=False, on_epoch=True, logger=True)

        return loss

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
            threshold=self.hparams.lr_scheduler_threshold
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc"
            }
        }

"""
import torch
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelF1Score
from models import RASFinderV2


class RASFinderV2Lightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters()

        self.model = RASFinderV2(
            input_dim=config["input_dim"],
            summary_dim=config["summary_dim"],
            dim_model=config["dim_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dim_feedforward=config["dim_feedforward"]
        )

        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.val_metric = MultilabelF1Score(num_labels=2)
        self.test_metric = MultilabelF1Score(num_labels=2)

    # =============================
    # forward
    # =============================
    def forward(self, site, summary):
        return self.model(site, summary)

    # =============================
    # common step
    # =============================
    def step(self, batch):
        site, summary, label = batch

        logits = self(site, summary)

        loss = self.loss_fn(logits, label.float())

        return loss, logits, label

    # =============================
    # training
    # =============================
    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    # =============================
    # validation
    # =============================
    def validation_step(self, batch, batch_idx):
        loss, logits, label = self.step(batch)

        probs = torch.sigmoid(logits)
        preds = probs > 0.5

        f1 = self.val_metric(preds.int(), label.int())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

    # =============================
    # 🔥 TEST STEP (tối ưu tốc độ)
    # =============================
    def test_step(self, batch, batch_idx):

        with torch.no_grad():
            site, summary, label = batch

            logits = self(site, summary)

            probs = torch.sigmoid(logits)

            preds = probs > 0.5

            loss = self.loss_fn(logits, label.float())

            f1 = self.test_metric(preds.int(), label.int())

            self.log("test_loss", loss, prog_bar=True)
            self.log("test_f1", f1, prog_bar=True)

        # return để trainer gom lại nếu cần
        return {
            "probs": probs.detach().cpu(),
            "preds": preds.detach().cpu(),
            "labels": label.detach().cpu()
        }

    # =============================
    # optimizer
    # =============================
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)
"""
