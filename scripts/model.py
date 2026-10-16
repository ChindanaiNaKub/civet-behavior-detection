"""The behavior classifier and its on-disk form.

The professor's Phase 2 Task 1 asks for a CNN over images. This is that CNN,
applied to each frame of a five-second window and pooled over time into one
prediction per window, which is what lets a still window and a moving window
differ. The backbone is ImageNet-pretrained because 22 source clips cannot train
a video network from scratch, and only the small head is trained from the data.

One checkpoint file holds both the weights and the configuration the weights
were trained with, so loading a model cannot silently disagree with the frame
count or class list it was trained on.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision

import frames

BACKBONE = "resnet18"
BACKBONE_FEATURES = 512


class BehaviorCNN(nn.Module):
    """A per-frame CNN with mean and max pooling over the window's frames."""

    def __init__(self, classes, dropout: float = 0.3):
        super().__init__()
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Identity()
        self.backbone = net
        # Mean pooling summarises the window; max pooling keeps a frame that only
        # occurs briefly, which is how a short action survives eight samples.
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(BACKBONE_FEATURES * 2, len(classes)))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        count, time = batch.shape[0], batch.shape[1]
        features = self.backbone(batch.flatten(0, 1)).view(count, time, -1)
        pooled = torch.cat([features.mean(dim=1), features.amax(dim=1)], dim=1)
        return self.head(pooled)


def save(path, model, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "state_dict": model.state_dict()}, path)


def load(path, device: str = "cuda"):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    config = blob["config"]
    model = BehaviorCNN(config["classes"])
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval(), config


def make_config(classes, extra: dict | None = None) -> dict:
    config = {
        "classes": list(classes),
        "backbone": BACKBONE,
        "samples": frames.SAMPLES,
        "wh": list(frames.WH),
    }
    config.update(extra or {})
    return config


def probabilities(model, decoded, indices, device, batch: int = 8) -> np.ndarray:
    """Class probabilities for each decoded window, shape (len(indices), classes)."""
    out = np.zeros((len(indices), len(model.head[-1].weight)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            chunk = list(indices[start : start + batch])
            batch_input = frames.to_tensor(decoded[chunk]).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(batch_input)
            out[start : start + len(chunk)] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out
