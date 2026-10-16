"""Train the frame-CNN behavior classifier on the verified windows.

Model settings, the number of epochs and the early-stopping decision all come
from the training and validation windows. evaluate.py owns the locked test
clips, so nothing here can tune against them (issue #1 evaluation rules).

Usage:
    python3 scripts/train.py --out outputs/model/model.pt
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import frames
import labels
import metrics
import model as M
import windows as W

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
BATCH = 8
EPOCHS = 40
PATIENCE = 10


def class_weights(y_true, classes) -> torch.Tensor:
    """Inverse frequency, so the most common behavior does not set the loss."""
    counts = np.array([sum(1 for label in y_true if label == c) for c in classes], dtype=np.float64)
    return torch.tensor(len(y_true) / (len(classes) * np.maximum(counts, 1.0)), dtype=torch.float32)


def train_epoch(model, decoded, indices, targets, optimizer, scaler, weights, device, batch: int = BATCH) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss(weight=weights)
    order = np.random.permutation(len(indices))
    total = 0.0
    for start in range(0, len(order), batch):
        chunk = order[start : start + batch]
        batch_input = frames.to_tensor(decoded[[indices[i] for i in chunk]]).to(device)
        batch_target = torch.tensor([targets[i] for i in chunk], device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = criterion(model(batch_input), batch_target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * len(chunk)
    return total / len(indices)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--windows", type=Path, default=Path("outputs/windows/windows.csv"))
    parser.add_argument("--source-clips", type=Path, default=Path("data/source-clips.csv"))
    parser.add_argument("--clips-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("outputs/model/model.pt"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classes", default=",".join(labels.MODEL_CLASSES), help="comma-separated classes this run covers")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    classes = [name.strip() for name in args.classes.split(",") if name.strip()]
    unknown = [name for name in classes if name not in labels.ALL_LABELS]
    if unknown:
        parser.error(f"not a known behavior: {unknown}; known are {list(labels.ALL_LABELS)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("CUDA is not available; running on CPU")

    all_windows = W.read_windows(args.windows)
    relative = W.read_relative_paths(args.source_clips)

    train_windows = [w for w in all_windows if w.split == "train"]
    val_windows = [w for w in all_windows if w.split == "validation"]
    if not train_windows or not val_windows:
        parser.error(f"need train and validation windows; have {len(train_windows)} and {len(val_windows)}")

    # data/raw/<relative_path> is where the tracked clips live (ADR 0002).
    train_frames = frames.decode_windows(train_windows, args.clips_root, relative)
    val_frames = frames.decode_windows(val_windows, args.clips_root, relative)
    train_labels = [w.behavior for w in train_windows]
    val_labels = [w.behavior for w in val_windows]
    missing = [c for c in classes if c not in set(train_labels)]
    if missing:
        parser.error(f"no training window for {missing}; a class the model never sees cannot be reported on")

    print(f"device {device} | train {len(train_frames)} windows {dict(Counter(train_labels))}")
    print(f"validation {len(val_frames)} windows {dict(Counter(val_labels))}")

    target_of = {name: index for index, name in enumerate(classes)}
    train_targets = [target_of[label] for label in train_labels]

    model = M.BehaviorCNN(classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    weights = class_weights(train_labels, classes).to(device)

    best = {"macro_f1": -1.0, "epoch": -1, "state": None}
    history = []
    stale = 0
    indices = np.arange(len(train_frames))
    val_indices = np.arange(len(val_frames))
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_frames, indices, train_targets, optimizer, scaler, weights, device, args.batch)
        probabilities = M.probabilities(model, val_frames, val_indices, device, args.batch)
        preds = [classes[i] for i in probabilities.argmax(axis=1)]
        val_macro_f1 = metrics.report(val_labels, preds, classes)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": round(loss, 4), "val_macro_f1": round(val_macro_f1, 4)})
        print(f"epoch {epoch:>3}  loss {loss:.4f}  val macro F1 {val_macro_f1:.4f}", flush=True)
        if val_macro_f1 > best["macro_f1"]:
            best = {"macro_f1": val_macro_f1, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"no validation gain for {PATIENCE} epochs; stopping")
                break

    model.load_state_dict(best["state"])
    config = M.make_config(
        classes,
        {
            "seed": args.seed,
            "batch": args.batch,
            "epochs_requested": args.epochs,
            "best_epoch": best["epoch"],
            "val_macro_f1": round(best["macro_f1"], 4),
            "val_windows": len(val_frames),
            "train_windows": len(train_frames),
            "history": history,
        },
    )
    M.save(args.out, model, config)
    print(f"\nbest validation macro F1 {best['macro_f1']:.4f} at epoch {best['epoch']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
