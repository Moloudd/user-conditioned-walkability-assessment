import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def corn_logits_to_predictions(logits, as_numpy=True):
    """
    Convert CORN logits to class predictions.

    CORN logits represent conditional probabilities P(y > k | y > k-1).
    The joint (unconditional) probability P(y > k) = prod_{i=0}^{k} sigmoid(logits[:, i]).
    The predicted class is the number of thresholds where this joint prob > 0.5.

    Parameters
    ----------
    logits : torch.Tensor or np.ndarray, shape (N, num_classes-1)
    as_numpy : bool
        If True, return numpy array; otherwise return torch.Tensor.

    Returns
    -------
    predictions : np.ndarray or torch.Tensor, shape (N,)

    Example
    -------
    >>> logits = torch.tensor([[14.152, -6.194, 0.477, 0.968],
    ...                        [65.667,  0.303, 11.50, -4.524]])
    >>> corn_logits_to_predictions(logits, as_numpy=False)
    tensor([1, 3])
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)          # joint cumulative probs
    predicted_labels = (probas > 0.5).sum(dim=1)  # count thresholds exceeded

    if as_numpy:
        return predicted_labels.cpu().numpy()
    return predicted_labels


def corn_logits_to_probs(logits, num_classes, as_numpy=True):
    """
    Convert CORN logits to class probabilities.

    Parameters
    ----------
    logits : torch.Tensor or np.ndarray, shape (N, num_classes-1)
    num_classes : int
    as_numpy : bool

    Returns
    -------
    probabilities : np.ndarray or torch.Tensor, shape (N, num_classes)
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    # Joint cumulative P(y > k) = prod_{i<=k} sigmoid(logit_i)
    cumulative_probs = torch.cumprod(torch.sigmoid(logits), dim=1)  # (N, num_classes-1)

    batch_size = logits.shape[0]
    probs = torch.zeros(batch_size, num_classes, dtype=logits.dtype, device=logits.device)

    # P(y = 0) = 1 - P(y > 0)
    probs[:, 0] = 1.0 - cumulative_probs[:, 0]
    # P(y = k) = P(y > k-1) - P(y > k)  for 1 <= k < num_classes-1
    for k in range(1, num_classes - 1):
        probs[:, k] = cumulative_probs[:, k - 1] - cumulative_probs[:, k]
    # P(y = num_classes-1) = P(y > num_classes-2)
    probs[:, num_classes - 1] = cumulative_probs[:, num_classes - 2]

    probs = torch.clamp(probs, min=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True)

    if as_numpy:
        return probs.cpu().numpy()
    return probs


class CornLoss(nn.Module):
    """
    CORN (Conditional Ordinal Regression for Neural networks) loss.

    Each of the num_classes-1 output logits represents the conditional
    probability P(y > k | y > k-1). The loss is a sum of binary
    cross-entropies computed only on the subset of examples eligible
    for each threshold, normalised by the total number of such examples.

    Reference
    ---------
    Shi et al. (2021) "Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities"

    Parameters
    ----------
    num_classes : int
        Total number of ordinal classes (labels start at 0).
    importance_weights : torch.Tensor, shape (num_classes-1,), optional
        Per-task loss weights. Uniform if None.

    Examples
    --------
    >>> import torch
    >>> _ = torch.manual_seed(0)
    >>> logits = torch.randn(8, 4)          # 5 classes → 4 logits
    >>> labels = torch.tensor([0,1,2,2,2,3,4,4])
    >>> loss_fn = CornLoss(num_classes=5)
    >>> loss_fn(logits, labels)             # scalar tensor
    tensor(...)
    """

    def __init__(
        self,
        num_classes: int,
        importance_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        if importance_weights is not None:
            iw = importance_weights.to(dtype=torch.float32).view(-1)
            self.register_buffer("importance_weights", iw)
        else:
            self.importance_weights = None

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor, shape (N, num_classes-1)
            Raw logits from the CORN head.
        target : torch.Tensor, shape (N,)
            Integer class labels in [0, num_classes-1].

        Returns
        -------
        loss : scalar torch.Tensor
        """
        logits = input
        labels = target.view(-1).long()

        total_loss = torch.zeros(1, dtype=logits.dtype, device=logits.device).squeeze()
        num_examples = 0

        for k in range(self.num_classes - 1):
            # Eligible examples: those with label >= k  (i.e. y > k-1)
            mask = labels >= k
            if mask.sum() < 1:
                continue

            binary_labels = (labels[mask] > k).to(logits.dtype)
            pred_k = logits[mask, k]

            per_example_loss = F.binary_cross_entropy_with_logits(
                pred_k, binary_labels, reduction="none"
            )

            # Apply per-task importance weight
            if self.importance_weights is not None:
                per_example_loss = per_example_loss * self.importance_weights[k]

            total_loss = total_loss + per_example_loss.sum()
            num_examples += mask.sum().item()

        if num_examples == 0:
            return torch.zeros(1, dtype=logits.dtype, device=logits.device).squeeze()

        return total_loss / num_examples
