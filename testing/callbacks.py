"""
Callbacks for testing scripts.

Collect predictions during model testing.
"""

import torch
import numpy as np
from pytorch_lightning.callbacks import Callback

class PredictionCollector(Callback):
    def __init__(self, top_k=3):
        super().__init__()
        self.top_k = top_k
        self.all_probs = []
        self.all_preds = []
        self.all_labels = []
        self.all_keys = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if len(batch) == 3:
            # QFinder: (features, labels)
            features, labels, keys = batch
            with torch.no_grad():
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
        else:
            # RHASFinder: (sitewise_features, summary_features, lengths, labels)
            sitewise_features, summary_features, lengths, labels, keys = batch
            with torch.no_grad():
                logits = outputs["logits"] #pl_module(sitewise_features, lengths, summary_features)
                labels = outputs["labels"]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

        self.all_probs.append(probs.cpu())
        self.all_preds.append(preds.cpu())
        self.all_labels.append(labels.cpu())
        self.all_keys.extend(keys)

    def get_results(self):
        y_prob = torch.cat(self.all_probs).float().cpu().numpy()
        y_pred = torch.cat(self.all_preds).numpy()
        y_true = torch.cat(self.all_labels).numpy()
        return y_true, y_pred, y_prob, self.all_keys
