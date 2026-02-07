"""
Apply before importing/loading pyannote models.
PyTorch 2.6+ uses weights_only=True by default; pyannote checkpoints contain
PyTorch Lightning classes → load fails. We trust HF checkpoints, so force weights_only=False.
"""
import torch

_torch_load = torch.load


def _load_trusted(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load(*args, **kwargs)


torch.load = _load_trusted
