"""The validation-only path must not decode a test window."""

from __future__ import annotations

import csv
import json
import sys

import numpy as np

import evaluate


def test_validation_only_decodes_only_validation_windows(monkeypatch, tmp_path):
    windows = tmp_path / "windows.csv"
    with windows.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "start_s", "end_s", "behavior", "split"])
        writer.writeheader()
        writer.writerow({"clip_id": "train", "start_s": 0, "end_s": 5, "behavior": "Eating", "split": "train"})
        writer.writerow({"clip_id": "val", "start_s": 0, "end_s": 5, "behavior": "Eating", "split": "validation"})
        writer.writerow({"clip_id": "test", "start_s": 0, "end_s": 5, "behavior": "Resting", "split": "test"})

    source = tmp_path / "source.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "relative_path"])
        writer.writeheader()
        for clip_id in ("train", "val", "test"):
            writer.writerow({"clip_id": clip_id, "relative_path": f"{clip_id}.mp4"})

    report = tmp_path / "validation.md"
    payload = tmp_path / "threshold.json"
    decoded = []

    monkeypatch.setattr(
        evaluate.M,
        "load",
        lambda path, device: (
            object(),
            {"classes": ["Eating", "Locomotion", "Resting"], "best_epoch": 1, "val_macro_f1": 1.0, "train_windows": 1},
        ),
    )
    monkeypatch.setattr(evaluate.frames, "decode_windows", lambda windows, clips_root, relative: decoded.extend(windows) or [[0]])
    monkeypatch.setattr(evaluate.M, "probabilities", lambda model, frames, indices, device: np.array([[0.9, 0.05, 0.05]]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--windows", str(windows),
            "--source-clips", str(source),
            "--model", str(tmp_path / "model.pt"),
            "--validation-only",
            "--report", str(report),
            "--json", str(payload),
            "--device", "cpu",
        ],
    )

    assert evaluate.main() == 0
    assert decoded and all(window.split == "validation" for window in decoded)
    assert json.loads(payload.read_text())["test"] is None
    assert "No test window was decoded or scored" in report.read_text()


def test_held_out_payload_keeps_every_test_window_traceable(monkeypatch, tmp_path):
    windows = tmp_path / "windows.csv"
    with windows.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "start_s", "end_s", "behavior", "split"])
        writer.writeheader()
        writer.writerow({"clip_id": "train", "start_s": 0, "end_s": 5, "behavior": "Eating", "split": "train"})
        writer.writerow({"clip_id": "val", "start_s": 0, "end_s": 5, "behavior": "Eating", "split": "validation"})
        writer.writerow({"clip_id": "val", "start_s": 5, "end_s": 10, "behavior": "Resting", "split": "validation"})
        writer.writerow({"clip_id": "test", "start_s": 0, "end_s": 5, "behavior": "Eating", "split": "test"})
        writer.writerow({"clip_id": "test", "start_s": 5, "end_s": 10, "behavior": "Resting", "split": "test"})

    source = tmp_path / "source.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "relative_path"])
        writer.writeheader()
        for clip_id in ("train", "val", "test"):
            writer.writerow({"clip_id": clip_id, "relative_path": f"{clip_id}.mp4"})

    report = tmp_path / "evaluation.md"
    payload = tmp_path / "evaluation.json"
    calls = []

    monkeypatch.setattr(
        evaluate.M,
        "load",
        lambda path, device: (
            object(),
            {"classes": ["Eating", "Locomotion", "Resting"], "best_epoch": 1, "val_macro_f1": 1.0, "train_windows": 1},
        ),
    )
    monkeypatch.setattr(evaluate.frames, "decode_windows", lambda windows, clips_root, relative: [[0]] * len(windows))

    def probabilities(model, decoded, indices, device):
        calls.append(len(list(indices)))
        if len(calls) == 1:
            return np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])
        return np.array([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]])

    monkeypatch.setattr(evaluate.M, "probabilities", probabilities)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--windows", str(windows),
            "--source-clips", str(source),
            "--model", str(tmp_path / "model.pt"),
            "--report", str(report),
            "--json", str(payload),
            "--device", "cpu",
        ],
    )

    assert evaluate.main() == 0
    scored = json.loads(payload.read_text())["windows"]
    assert [(row["clip_id"], row["start_s"], row["behavior"], row["predicted"]) for row in scored] == [
        ("test", 0.0, "Eating", "Eating"),
        ("test", 5.0, "Resting", "Resting"),
    ]
    assert all("probabilities" in row and "confidence" in row for row in scored)
