import torch.nn as nn

try:
    from .checkpoint_utils import load_checkpoint
    from .model import LiteMultiBeamModel
except ImportError:
    from checkpoint_utils import load_checkpoint
    from model import LiteMultiBeamModel


BRANCH = "uav_numeric"


class UAVSideNumericBeamPrediction(nn.Module):
    """UAV-side numeric-only beam prediction model.

    Input:
        numeric: [B, 10, 5]
    Output:
        logits: [B, 5, 64]
    """

    def __init__(self):
        super().__init__()
        self.model = LiteMultiBeamModel(
            numeric_use_freq_branch=True,
            numeric_use_aux_branch=True,
        )

    def forward(self, numeric):
        return self.model(numeric=numeric, mode="numeric")


def build_model(device=None):
    model = UAVSideNumericBeamPrediction()
    if device is not None:
        model = model.to(device)
    return model


def load_weights(model, weight_path=None, map_location="cpu", strict=None):
    return load_checkpoint(model, BRANCH, weight_path=weight_path, map_location=map_location, strict=strict)
