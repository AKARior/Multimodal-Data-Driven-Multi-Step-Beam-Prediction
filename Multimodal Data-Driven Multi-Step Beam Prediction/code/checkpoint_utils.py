from pathlib import Path

import torch


BRANCH_ALIASES = {
    "numeric": "uav_numeric",
    "uav": "uav_numeric",
    "uav_numeric": "uav_numeric",
    "uav_side_numeric": "uav_numeric",
    "image": "edge_image",
    "edge_image": "edge_image",
    "edge_side_image": "edge_image",
    "multi": "edge_multimodal",
    "multimodal": "edge_multimodal",
    "edge_multimodal": "edge_multimodal",
    "edge_multimodal_fusion": "edge_multimodal",
}


WEIGHT_FILES = {
    "uav_numeric": "uav_side_numeric_beam_prediction_ckpt.pth",
    "edge_image": "edge_side_image_beam_prediction_ckpt.pth",
    "edge_multimodal": "edge_multimodal_beam_prediction_ckpt.pth",
}


TRAINING_INFO = {
    "uav_numeric": {
        "display_name": "UAV-side numeric beam prediction",
        "training_epochs": 200,
        "note": "Checkpoint from the numeric pretraining stage.",
    },
    "edge_image": {
        "display_name": "Edge-side image beam prediction",
        "training_epochs": 100,
        "note": "Checkpoint from the 100-epoch image pretraining stage.",
    },
    "edge_multimodal": {
        "display_name": "Edge multimodal fusion beam prediction",
        "training_epochs": 50,
        "note": "Checkpoint from the 50-epoch no-dominant adaptive fusion stage.",
    },
}


def normalize_branch(branch):
    key = str(branch).lower().replace("-", "_").replace(" ", "_")
    if key not in BRANCH_ALIASES:
        valid = ", ".join(sorted(BRANCH_ALIASES))
        raise ValueError(f"Unknown branch '{branch}'. Valid aliases: {valid}")
    return BRANCH_ALIASES[key]


def archive_root():
    return Path(__file__).resolve().parents[1]


def checkpoint_path(branch, root=None):
    branch = normalize_branch(branch)
    base = Path(root) if root is not None else archive_root()
    return base / "weights" / WEIGHT_FILES[branch]


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        return checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    return checkpoint


def unwrap_model(model):
    return getattr(model, "model", model)


def filter_state_dict_for_branch(model, state_dict, branch):
    branch = normalize_branch(branch)
    target = unwrap_model(model)
    model_state = target.state_dict()

    if branch == "uav_numeric":
        prefixes = ("numeric.", "decoder.")
    elif branch == "edge_image":
        prefixes = ("vision.", "decoder.")
    elif branch == "edge_multimodal":
        prefixes = None
    else:
        raise ValueError(f"Unknown branch: {branch}")

    if prefixes is None:
        return state_dict

    filtered = {}
    for key, value in state_dict.items():
        if key.startswith(prefixes) and key in model_state and model_state[key].shape == value.shape:
            filtered[key] = value
    return filtered


def load_checkpoint(model, branch, weight_path=None, root=None, map_location="cpu", strict=None):
    branch = normalize_branch(branch)
    if strict is None:
        strict = branch == "edge_multimodal"
    ckpt_path = Path(weight_path) if weight_path is not None else checkpoint_path(branch, root=root)
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    state_dict = filter_state_dict_for_branch(model, extract_state_dict(checkpoint), branch)
    target = unwrap_model(model)
    target.load_state_dict(state_dict, strict=strict)
    return checkpoint
