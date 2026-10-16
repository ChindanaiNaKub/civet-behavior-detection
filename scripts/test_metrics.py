"""Self-check for the metric and threshold logic. Pure numpy: `python -m pytest scripts`."""

import numpy as np
import pytest

import metrics as M

CLASSES = ("A", "B", "C")


def test_confusion_places_abstentions_in_their_own_column():
    matrix = M.confusion(["A", "A", "B", "B", "C"], ["A", "B", "B", None, "C"], CLASSES)
    assert matrix == [[1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0]]


def test_report_matches_hand_computed_values():
    result = M.report(["A", "A", "B", "B", "C"], ["A", "B", "B", None, "C"], CLASSES)
    assert result["support"] == {"A": 2, "B": 2, "C": 1}
    assert result["precision"]["A"] == pytest.approx(1.0)   # 1 tp / 1 predicted A
    assert result["recall"]["A"] == pytest.approx(0.5)      # 1 tp / 2 true A
    assert result["precision"]["B"] == pytest.approx(0.5)   # 1 tp / 2 predicted B
    assert result["f1"]["A"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.5 + 1.0) / 3)
    assert result["abstain_rate"] == pytest.approx(0.2)
    assert result["accuracy_decided"] == pytest.approx(0.75)


def test_abstaining_everywhere_scores_zero_not_high():
    result = M.report(["A", "B", "A"], [None, None, None], CLASSES)
    assert result["macro_f1"] == 0.0
    assert result["abstain_rate"] == 1.0
    assert result["accuracy_decided"] == 0.0


def test_a_class_with_no_test_example_is_named_and_excluded_from_the_supported_macro():
    result = M.report(["A", "A"], ["A", "A"], CLASSES)
    assert result["unsupported_classes"] == ["B", "C"]
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["macro_f1_supported"] == pytest.approx(1.0)


def test_unknown_labels_are_rejected():
    with pytest.raises(ValueError, match="unknown true label"):
        M.confusion(["Z"], ["A"], CLASSES)
    with pytest.raises(ValueError, match="unknown predicted label"):
        M.confusion(["A"], ["Z"], CLASSES)


def test_best_threshold_abstains_on_the_unconfident_window():
    probabilities = np.array([[0.90, 0.05], [0.40, 0.35]])
    threshold, score = M.best_threshold(probabilities, ["A", "B"], ("A", "B"))
    assert threshold > 0.4
    assert score == pytest.approx(0.5)


def test_best_threshold_keeps_the_threshold_at_zero_when_every_call_is_confident():
    probabilities = np.array([[0.99, 0.005], [0.005, 0.99]])
    threshold, score = M.best_threshold(probabilities, ["A", "B"], ("A", "B"))
    assert threshold == 0.0
    assert score == pytest.approx(1.0)
