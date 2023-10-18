import time

import torch
import torch.nn.functional as F


def cross_entropy(logits, labels):
    """Compute cross entropy loss for any shape

    Args:
        logits: [*, num_classes]
        labels: [*]

    Returns:
        [*]
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction='none'
    ).view(*labels.shape)


def angle_loss(logits, labels, min_norm=0.1):
    """Compute angle loss for any shape

    Args:
        logits: [*, 2]
        labels: (cos, sin) [*, 2]

    Returns:
        [*]
    """
    norm = torch.clamp(logits.norm(dim=-1, keepdim=True), min=min_norm)
    logits = logits / norm
    dot = torch.einsum('...d,...d->...', logits, labels)
    return 1. - dot


class Timer:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed_time = time.perf_counter() - self._start
        print(self.text.format(elapsed_time))
