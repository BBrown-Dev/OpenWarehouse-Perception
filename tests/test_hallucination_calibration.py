import numpy as np
import pandas as pd
import pytest

from openwarehouse.evaluation.calibration import calibration_errors, reliability_table
from openwarehouse.evaluation.hallucination import chair, pope_questions, pope_scores
from openwarehouse.evaluation.metrics import localization_iou


def frame(rows):
    return pd.DataFrame(rows, columns=["image_id", "class_name", "is_positive", "says_yes"])


def test_chair_counts_only_verified_negatives_as_hallucinations():
    decisions = frame(
        [
            ("a", "box", True, True),
            ("a", "truck", False, True),  # hallucinated
            ("b", "box", True, True),
            ("b", "truck", False, False),
            ("c", "box", False, False),
        ]
    )
    result = chair(decisions)
    assert result["chair_i"] == pytest.approx(1 / 3)
    assert result["chair_s"] == pytest.approx(1 / 3)


def test_pope_questions_are_balanced_and_use_only_verified_labels():
    labels = pd.DataFrame(
        [
            ("a", "person", True), ("a", "box", True), ("a", "truck", False), ("a", "ladder", False),
            ("b", "person", True), ("b", "truck", False),
            ("c", "box", True),  # no verified negative: contributes nothing
        ],
        columns=["image_id", "class_name", "is_positive"],
    )
    for strategy in ("random", "popular", "adversarial"):
        questions = pope_questions(labels, strategy, seed=0)
        assert questions["is_positive"].sum() == (~questions["is_positive"]).sum() == 3
        assert "c" not in set(questions["image_id"])
        merged = questions.merge(labels, on=["image_id", "class_name"], suffixes=("", "_label"))
        assert (merged["is_positive"] == merged["is_positive_label"]).all()


def test_pope_adversarial_picks_the_most_co_occurring_absent_class():
    labels = pd.DataFrame(
        [
            ("x", "person", True), ("x", "hard hat", True),
            ("y", "person", True), ("y", "hard hat", True),
            ("z", "person", True), ("z", "hard hat", False), ("z", "giraffe", False),
        ],
        columns=["image_id", "class_name", "is_positive"],
    )
    questions = pope_questions(labels, "adversarial", seed=0, per_image=1)
    negative = questions[(questions["image_id"] == "z") & ~questions["is_positive"]]
    assert negative["class_name"].tolist() == ["hard hat"]


def test_pope_scores():
    answers = frame([("a", "p", True, True), ("a", "q", False, True), ("b", "p", True, False), ("b", "q", False, False)])
    result = pope_scores(answers)
    assert result == {"questions": 4, "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "yes_ratio": 0.5}


def test_ece_is_zero_when_confidence_matches_frequency_and_large_when_overconfident():
    y = np.array([1] * 8 + [0] * 2)
    assert calibration_errors(y, np.full(10, 0.8))["ece"] == pytest.approx(0.0)
    overconfident = calibration_errors(np.array([1, 0, 1, 0]), np.full(4, 0.99))
    assert overconfident["ece"] == pytest.approx(0.49)
    assert overconfident["brier"] == pytest.approx((0.01**2 + 0.99**2) / 2)


def test_reliability_counts_a_confidence_of_one():
    table = reliability_table([1, 0], [1.0, 0.0], bins=10)
    assert table["count"].sum() == 2 and table["count"].iloc[-1] == 1


def test_localization_iou_takes_best_detection_per_target():
    targets = [[0, 0, 10, 10], [20, 20, 30, 30]]
    detections = [[0, 0, 10, 5], [20, 20, 30, 30]]
    assert localization_iou(detections, targets).tolist() == pytest.approx([0.5, 1.0])
    assert localization_iou([], targets).tolist() == [0.0, 0.0]


def test_localization_iou_is_one_to_one():
    # One detection overlapping two people may only be credited to one of them.
    targets = [[0, 0, 10, 10], [0, 0, 10, 10]]
    assert sorted(localization_iou([[0, 0, 10, 10]], targets).tolist()) == [0.0, 1.0]


def test_wilson_interval():
    import sys
    sys.path.insert(0, "scripts")
    from compute_safety_metrics import wilson

    low, high = wilson(0, 300)
    assert low == 0.0 and 0.0 < high < 0.013
    low, high = wilson(12, 14)
    assert round(low, 2) == 0.60 and round(high, 2) == 0.96


def test_match_one_to_one_returns_each_detection_at_most_once():
    from openwarehouse.evaluation.metrics import match_one_to_one

    targets = [[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]]
    detections = [[0, 0, 10, 10], [50, 50, 60, 60]]
    matches = match_one_to_one(detections, targets)
    used = [detection for detection, _ in matches.values()]
    assert len(used) == len(set(used)) == 2
    assert matches[0][0] == 0 and matches[2] == (1, 1.0) and 1 not in matches
