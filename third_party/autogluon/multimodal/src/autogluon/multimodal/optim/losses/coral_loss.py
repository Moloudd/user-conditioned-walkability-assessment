import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def coral_logits_to_predictions(logits, as_numpy=True):
    """
    Convert CORAL cumulative logits to class predictions.
    
    For CORAL, we have num_classes-1 logits representing P(y > k) for k=0..num_classes-2.
    The predicted class is the sum of probabilities: sum(sigmoid(logits)).
    
    Parameters
    ----------
    logits : torch.Tensor or np.ndarray
        Logits of shape (N, num_classes-1) from CORAL head
    as_numpy : bool
        If True, return numpy array. If False, return torch.Tensor
        
    Returns
    -------
    predictions : np.ndarray or torch.Tensor
        Class predictions of shape (N,)
        
    Example
    -------
    For 5 classes (logits shape = (N, 4)):
        logits = [[2.0, 1.0, -1.0, -3.0]]  # High prob for first thresholds
        sigmoid(logits) ≈ [0.88, 0.73, 0.27, 0.05]
        sum ≈ 1.93 → round to 2 (class 2)
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    
    # Apply sigmoid to get cumulative probabilities
    cumulative_probs = torch.sigmoid(logits)
    
    # Sum across thresholds to get expected class
    predictions = cumulative_probs.sum(dim=1)
    
    # Round to nearest integer to get discrete class
    predictions = torch.round(predictions).long()
    
    if as_numpy:
        return predictions.cpu().numpy()
    return predictions


def coral_logits_to_probs(logits, num_classes, as_numpy=True):
    """
    Convert CORAL cumulative logits to class probabilities.
    
    Parameters
    ----------
    logits : torch.Tensor or np.ndarray
        Logits of shape (N, num_classes-1) from CORAL head
    num_classes : int
        Total number of classes
    as_numpy : bool
        If True, return numpy array. If False, return torch.Tensor
        
    Returns
    -------
    probabilities : np.ndarray or torch.Tensor
        Class probabilities of shape (N, num_classes)
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    
    # Apply sigmoid to get P(y > k)
    cumulative_probs = torch.sigmoid(logits)  # (N, num_classes-1)
    
    # Compute class probabilities from cumulative probabilities
    # P(y = 0) = 1 - P(y > 0)
    # P(y = k) = P(y > k-1) - P(y > k) for k > 0
    # P(y = num_classes-1) = P(y > num_classes-2)
    
    batch_size = logits.shape[0]
    probs = torch.zeros(batch_size, num_classes, dtype=logits.dtype, device=logits.device)
    
    # First class: 1 - P(y > 0)
    probs[:, 0] = 1.0 - cumulative_probs[:, 0]
    
    # Middle classes: P(y > k-1) - P(y > k)
    for k in range(1, num_classes - 1):
        probs[:, k] = cumulative_probs[:, k-1] - cumulative_probs[:, k]
    
    # Last class: P(y > num_classes-2)
    probs[:, num_classes-1] = cumulative_probs[:, num_classes-2]
    
    # Clamp to ensure valid probabilities
    probs = torch.clamp(probs, min=0.0, max=1.0)
    
    # Normalize to ensure sum to 1 (numerical stability)
    probs = probs / probs.sum(dim=1, keepdim=True)
    
    if as_numpy:
        return probs.cpu().numpy()
    return probs


def labels_to_coral_levels(
    labels: torch.Tensor,
    num_classes: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert integer class labels to CORAL extended-binary levels.

    labels: shape (N,) with integer labels in [0, num_classes-1]
    returns: shape (N, num_classes-1) with 1s up to the label index, then 0s.

    Example (num_classes=5):
        label=0 -> [0, 0, 0, 0]
        label=1 -> [1, 0, 0, 0]
        label=3 -> [1, 1, 1, 0]
        label=4 -> [1, 1, 1, 1]
    """
    if labels.dim() != 1:
        labels = labels.view(-1)

    if (labels < 0).any() or (labels > (num_classes - 1)).any():
        raise ValueError(
            f"Labels must be in [0, {num_classes-1}], got range "
            f"[{labels.min().item()}, {labels.max().item()}]"
        )

    device = labels.device
    # shape: (1, num_classes-1): 0,1,2,...,K-2
    thresholds = torch.arange(num_classes - 1, device=device).unsqueeze(0)
    # shape: (N, 1)
    labels_expanded = labels.unsqueeze(1)

    # For CORAL: first `label` positions are 1, rest 0
    levels = (thresholds < labels_expanded).to(dtype)
    return levels


class CoralLoss(nn.Module):
    """
    CORAL (Cumulative Ordinal Regression with Logistic) loss.

    Expects logits of shape (N, num_classes-1).
    Targets can be:
      - integer class labels of shape (N,)
      - or already CORAL levels of shape (N, num_classes-1)
    """

    def __init__(
        self,
        num_classes: int,
        reduction: str = "mean",
        importance_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

        if importance_weights is not None:
            # importance_weights should be shape (num_classes-1,)
            iw = importance_weights.to(dtype=torch.float32).view(1, -1)
            self.register_buffer("importance_weights", iw)
        else:
            self.importance_weights = None

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input:  (N, num_classes-1) – raw logits
        target: (N,) int labels 0..num_classes-1  OR levels (N, num_classes-1)
        """
        logits = input

        if target.dim() == 1 or target.size(-1) == 1:
            # Standard AutoGluon labels: 1D, integer
            levels = labels_to_coral_levels(
                labels=target.view(-1),
                num_classes=self.num_classes,
                dtype=logits.dtype,
            )
        else:
            # Already in levels format
            levels = target.to(dtype=logits.dtype)

        levels = levels.to(device=logits.device)

        if logits.shape != levels.shape:
            raise ValueError(
                f"Logits shape {logits.shape} and levels shape {levels.shape} must match."
            )

        bce_loss = F.binary_cross_entropy_with_logits(logits, levels, reduction="none")

        if self.importance_weights is not None:
            bce_loss = bce_loss * self.importance_weights  # broadcast on dim=1

        # Sum over levels → per-sample loss
        loss_per_sample = bce_loss.sum(dim=1)

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        elif self.reduction in (None, "none"):
            return loss_per_sample
        else:
            raise ValueError(
                f"Invalid reduction '{self.reduction}'. "
                f"Expected 'mean', 'sum', 'none', or None."
            )
