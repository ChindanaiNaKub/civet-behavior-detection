"""Class metrics that count an abstention against the score.

An abstention is a prediction of None, which the classifier emits when its
confidence falls below the Uncertain threshold. An abstention counts as a miss
for its true class's recall and enters no class's precision denominator, so
abstaining everywhere scores 0 instead of scoring well. Issue #5 requires that
treatment and issue #6 requires the Uncertain rate be reported next to it.

Macro F1 is averaged over the classes named here. When a class has no test
example its F1 is 0 and drags the macro down, so `macro_f1_supported` (classes
with at least one example) is reported alongside it and the per-class support is
always printed.
"""

from __future__ import annotations

import numpy as np

ABSTAIN = None


def confusion(y_true, y_pred, classes) -> list[list[int]]:
    """Rows are true classes, columns are predicted classes, last column abstains."""
    index = {c: i for i, c in enumerate(classes)}
    matrix = [[0] * (len(classes) + 1) for _ in classes]
    for true, pred in zip(y_true, y_pred):
        if true not in index:
            raise ValueError(f"unknown true label {true!r}")
        row = matrix[index[true]]
        if pred is ABSTAIN:
            row[-1] += 1
        elif pred in index:
            row[index[pred]] += 1
        else:
            raise ValueError(f"unknown predicted label {pred!r}")
    return matrix


def report(y_true, y_pred, classes) -> dict:
    matrix = confusion(y_true, y_pred, classes)
    support = {c: sum(matrix[i]) for i, c in enumerate(classes)}
    predicted = {c: sum(matrix[i][j] for i in range(len(classes))) for j, c in enumerate(classes)}
    total_abstain = sum(row[-1] for row in matrix)

    precision, recall, f1 = {}, {}, {}
    for i, c in enumerate(classes):
        tp = matrix[i][i]
        precision[c] = tp / predicted[c] if predicted[c] else 0.0
        recall[c] = tp / support[c] if support[c] else 0.0
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c]) if precision[c] + recall[c] else 0.0

    supported = [c for c in classes if support[c]]
    correct = sum(matrix[i][i] for i in range(len(classes)))
    decided = len(y_true) - total_abstain
    return {
        "classes": list(classes),
        "matrix": matrix,
        "support": support,
        "predicted": predicted,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": sum(f1.values()) / len(classes) if classes else 0.0,
        "macro_f1_supported": sum(f1[c] for c in supported) / len(supported) if supported else 0.0,
        "accuracy_decided": correct / decided if decided else 0.0,
        "abstain_rate": total_abstain / len(y_true) if y_true else 0.0,
        "n": len(y_true),
        "n_abstain": total_abstain,
        "unsupported_classes": [c for c in classes if not support[c]],
    }


def markdown(metrics: dict, title: str) -> str:
    """The report block, so the numbers in docs never drift from the run."""
    lines = [
        f"### {title}",
        "",
        f"Windows: {metrics['n']}. Abstained: {metrics['n_abstain']} "
        f"({metrics['abstain_rate']:.1%}). Accuracy over decided windows: {metrics['accuracy_decided']:.3f}.",
        "",
        f"| Behavior | Support | Predicted | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in metrics["classes"]:
        lines.append(
            f"| {c} | {metrics['support'][c]} | {metrics['predicted'][c]} | "
            f"{metrics['precision'][c]:.3f} | {metrics['recall'][c]:.3f} | {metrics['f1'][c]:.3f} |"
        )
    lines += [
        "",
        f"**Macro F1 {metrics['macro_f1']:.3f}** over {len(metrics['classes'])} classes. "
        f"Macro F1 over the classes with test examples: {metrics['macro_f1_supported']:.3f}.",
    ]
    if metrics["unsupported_classes"]:
        lines.append(
            "No test example exists for: " + ", ".join(metrics["unsupported_classes"]) + ", so their F1 is 0 above."
        )
    lines += ["", "Confusion matrix, rows true, columns predicted, last column abstained:", "", "| true \\ pred | " 
              + " | ".join(metrics["classes"]) + " | abstain |", "|---|" + "---:|" * (len(metrics["classes"]) + 1)]
    for i, c in enumerate(metrics["classes"]):
        lines.append(f"| {c} | " + " | ".join(str(v) for v in metrics["matrix"][i]) + " |")
    return "\n".join(lines)


def best_threshold(probabilities, y_true, classes, grid=None):
    """The abstain threshold with the best macro F1 on validation data.

    Ties go to the lower threshold, which decides more windows at the same score.
    Call this on validation windows only; issue #1 forbids choosing the threshold
    on the test set.
    """
    grid = np.arange(0.0, 0.96, 0.05) if grid is None else grid
    classes = list(classes)
    best = (float(grid[0]), -1.0)
    for threshold in grid:
        top = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        preds = [classes[i] if confidence[j] >= threshold else ABSTAIN for j, i in enumerate(top)]
        score = report(y_true, preds, classes)["macro_f1"]
        if score > best[1] or (score == best[1] and threshold < best[0]):
            best = (float(threshold), score)
    return best
