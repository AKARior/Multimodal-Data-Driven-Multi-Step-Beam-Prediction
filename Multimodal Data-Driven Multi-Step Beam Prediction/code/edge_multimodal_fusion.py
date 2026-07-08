import torch.nn as nn

try:
    from .checkpoint_utils import load_checkpoint
    from .model import LiteMultiBeamModel
except ImportError:
    from checkpoint_utils import load_checkpoint
    from model import LiteMultiBeamModel


BRANCH = "edge_multimodal"


class EdgeMultimodalFusionBeamPrediction(nn.Module):
    """Edge-side multimodal beam prediction model.

    Inputs:
        numeric: [B, 10, 5]
        image: [B, 10, 3, H, W]
    Output:
        logits: [B, 5, 64]
    """

    def __init__(self):
        super().__init__()
        self.model = LiteMultiBeamModel(
            numeric_use_freq_branch=True,
            numeric_use_aux_branch=True,
            image_pooling_type="gem",
            image_residual_mode="mhc",
            image_use_sub_layer_norm=False,
            fusion_use_diff_term=True,
            fusion_use_prod_term=True,
            fusion_anchor_mode="adaptive",
            fusion_relation_only=True,
            fusion_normalize_relation_terms=True,
            fusion_diff_mode="abs",
        )

    def forward(self, numeric, image):
        return self.model(numeric=numeric, image=image, mode="multi")


def build_model(device=None):
    model = EdgeMultimodalFusionBeamPrediction()
    if device is not None:
        model = model.to(device)
    return model


def load_weights(model, weight_path=None, map_location="cpu", strict=None):
    return load_checkpoint(model, BRANCH, weight_path=weight_path, map_location=map_location, strict=strict)
