"""Score the trained classifier on the validation and locked test clips.

The abstain threshold is chosen on the validation windows and then applied to the
test windows once, which is the rule in issue #1: a threshold picked on test data
would make the reported number a tuning artifact. Issue #6 asks for macro F1,
per-class precision/recall/F1, a confusion matrix including abstentions, and the
abstain rate; all four come out of metrics.py, and any class the test clips
cannot support is named in the output rather than folded into the aggregate.

``--validation-only`` chooses the threshold and stops before the test split is
decoded, so issue #5's demo can run before the test set is locked. It writes
outputs/validation.md and outputs/threshold.json by default, never the held-out
report paths.

Writes the markdown report for docs/EVALUATION-REPORT.md and the JSON that
scripts/demo.py reads for the chosen threshold.

Usage:
    python3 scripts/evaluate.py --model outputs/model/model.pt
    python3 scripts/evaluate.py --validation-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import frames
import metrics
import model as M
import windows as W


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--windows", type=Path, default=Path("outputs/windows/windows.csv"))
    parser.add_argument("--source-clips", type=Path, default=Path("data/source-clips.csv"))
    parser.add_argument("--clips-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--model", type=Path, default=Path("outputs/model/model.pt"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--validation-only", action="store_true", help="choose the threshold on validation, never decode or score the test split")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.validation_only:
        args.report = args.report or Path("outputs/validation.md")
        args.json = args.json or Path("outputs/threshold.json")
    else:
        args.report = args.report or Path("outputs/evaluation.md")
        args.json = args.json or Path("outputs/evaluation.json")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, config = M.load(args.model, str(device))
    classes = config["classes"]

    all_windows = W.read_windows(args.windows)
    relative = W.read_relative_paths(args.source_clips)
    val_windows = [w for w in all_windows if w.split == "validation"]
    test_windows = [w for w in all_windows if w.split == "test"]
    if not val_windows:
        parser.error("need validation windows; have 0")
    if not test_windows and not args.validation_only:
        parser.error(f"need test windows; have {len(test_windows)}")

    val_labels = [w.behavior for w in val_windows]
    val_probabilities = M.probabilities(model, frames.decode_windows(val_windows, args.clips_root, relative), range(len(val_windows)), device)

    threshold, val_score = metrics.best_threshold(val_probabilities, val_labels, classes)
    val_preds = [classes[i] if val_probabilities[j].max() >= threshold else None for j, i in enumerate(val_probabilities.argmax(axis=1))]
    validation = metrics.report(val_labels, val_preds, classes)

    if args.validation_only:
        lines = [
            "# Validation-only threshold selection",
            "",
            f"Model `{args.model}` trained to epoch {config['best_epoch']} with validation macro F1 {config['val_macro_f1']} "
            f"over {config['train_windows']} training windows.",
            f"Abstain threshold {threshold:.2f}, chosen on the {len(val_windows)} validation windows "
            f"(validation macro F1 {val_score:.3f} at that threshold, {validation['abstain_rate']:.1%} abstained).",
            "No test window was decoded or scored. This is not a held-out result; run this script without"
            " `--validation-only` after the test set is locked.",
            "",
            metrics.markdown(validation, "Validation clips"),
            "",
            "## Reading this honestly",
            "",
            "- The threshold in this report is the one `scripts/demo.py` should use until the test set is locked.",
            "- Validation support is small. Read each class's support beside its score.",
            "- Run `python3 scripts/evaluate.py --validation-only` again to reproduce; the split is fixed in data/splits.csv.",
        ]
        report = "\n".join(lines)
        payload = {"mode": "validation-only", "threshold": threshold, "validation": validation, "test": None, "config": config}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report + "\n")
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(report)
        print(f"\nwritten to {args.report} and {args.json}")
        return 0

    test_labels = [w.behavior for w in test_windows]
    test_probabilities = M.probabilities(model, frames.decode_windows(test_windows, args.clips_root, relative), range(len(test_windows)), device)
    test_preds = [classes[i] if test_probabilities[j].max() >= threshold else None for j, i in enumerate(test_probabilities.argmax(axis=1))]
    test = metrics.report(test_labels, test_preds, classes)

    lines = [
        "# Held-out evaluation",
        "",
        f"Model `{args.model}` trained to epoch {config['best_epoch']} with validation macro F1 {config['val_macro_f1']} "
        f"over {config['train_windows']} training windows.",
        f"Abstain threshold {threshold:.2f}, chosen on the {len(val_windows)} validation windows "
        f"(validation macro F1 {val_score:.3f} at that threshold, {validation['abstain_rate']:.1%} abstained).",
        "The test clips were decoded and scored once, after the threshold was fixed.",
        "",
        metrics.markdown(test, "Test clips"),
        "",
        metrics.markdown(validation, "Validation clips, at the same threshold"),
        "",
        "## Reading this honestly",
        "",
        f"- The test split holds {test['n']} windows. Macro F1 over {len(classes)} classes is {test['macro_f1']:.3f}.",
        "- Macro F1 averages over every class, so a class with no test example contributes 0. The per-class support column shows where that happens.",
        "- Abstentions are counted as misses in recall and as no false positive in precision, so abstaining cannot raise the score.",
        "- Run `python3 scripts/evaluate.py` again to reproduce; the split is fixed in data/splits.csv.",
    ]
    report = "\n".join(lines)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"threshold": threshold, "validation": validation, "test": test, "config": config}, indent=2) + "\n")
    print(report)
    print(f"\nwritten to {args.report} and {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
