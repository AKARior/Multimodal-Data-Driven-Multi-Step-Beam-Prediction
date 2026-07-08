import torch.nn as nn

try:
    from .checkpoint_utils import load_checkpoint
    from .model import LiteMultiBeamModel
except ImportError:
    from checkpoint_utils import load_checkpoint
    from model import LiteMultiBeamModel


BRANCH = "edge_image"


class EdgeSideImageBeamPrediction(nn.Module):
    """Edge-side image-only beam prediction model.

    Input:
        image: [B, 10, 3, H, W]
    Output:
        logits: [B, 5, 64]
    """

    def __init__(self):
        super().__init__()
        self.model = LiteMultiBeamModel(
            image_pooling_type="gem",
            image_residual_mode="mhc",
            image_use_sub_layer_norm=False,
        )

    def forward(self, image):
        return self.model(image=image, mode="image")


def build_model(device=None):
    model = EdgeSideImageBeamPrediction()
    if device is not None:
        model = model.to(device)
    return model


def load_weights(model, weight_path=None, map_location="cpu", strict=None):
    return load_checkpoint(model, BRANCH, weight_path=weight_path, map_location=map_location, strict=strict)
