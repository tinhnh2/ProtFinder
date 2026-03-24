"""
Callbacks for testing scripts.

Collect predictions during model testing.
"""

import torch
import numpy as np
from pytorch_lightning.callbacks import Callback


class PredictionCollector_bk(Callback):
    """
    Collect all predictions from test batches for detailed analysis.
    """
    
    def __init__(self):
        super().__init__()
        self.all_predictions = []
        self.all_labels = []
    

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Collect predictions after each test batch.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module
            outputs: Outputs from test_step
            batch: Test batch
            batch_idx: Batch index
            dataloader_idx: DataLoader index
        """

        # For QFinder: batch is (features, labels)
        # For RASFinder: batch is (sitewise_features, summary_features, lengths, labels)
        if len(batch) == 2:
            # QFinder: (features, labels)
            features, labels = batch
            with torch.no_grad():
                logits = pl_module(features)
                predictions = torch.argmax(logits, dim=1)
        else:
            # RASFinder: (sitewise_features, summary_features, lengths, labels)
            sitewise_features, summary_features, lengths, labels = batch
            with torch.no_grad():
                logits = pl_module(sitewise_features, lengths, summary_features)
                predictions = torch.argmax(logits, dim=1)
        
        # Collect results
        self.all_predictions.extend(predictions.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
    

    def get_results(self):
        """
        Get collected results.
        
        Returns:
            Tuple of (predictions, labels) as numpy arrays
        """
        return (
            np.array(self.all_predictions),
            np.array(self.all_labels)
        )

class PredictionCollector(Callback):
    def __init__(self, top_k=3):
        super().__init__()
        self.top_k = top_k
        self.all_probs = []
        self.all_preds = []
        self.all_labels = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if len(batch) == 2:
            # QFinder: (features, labels)
            features, labels = batch
            with torch.no_grad():
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
        else:
            # RASFinder: (sitewise_features, summary_features, lengths, labels)
            sitewise_features, summary_features, lengths, labels = batch
            with torch.no_grad():
                logits = pl_module(sitewise_features, lengths, summary_features)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

        self.all_probs.append(probs.cpu())
        self.all_preds.append(preds.cpu())
        self.all_labels.append(labels.cpu())

    def get_results(self):
        y_prob = torch.cat(self.all_probs).numpy()
        y_pred = torch.cat(self.all_preds).numpy()
        y_true = torch.cat(self.all_labels).numpy()
        return y_true, y_pred, y_prob
