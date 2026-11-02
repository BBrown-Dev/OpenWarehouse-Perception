import math

import numpy as np

from openwarehouse.data.open_images import expand_box_labels
from openwarehouse.evaluation.metrics import auroc, bootstrap_ci, box_iou, detection_ap, mask_iou

BOX = [0.0, 0.0, 10.0, 10.0]
FAR = [50.0, 50.0, 60.0, 60.0]


def test_box_iou_half_overlap():
    iou = box_iou(np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([[5.0, 0.0, 15.0, 10.0]]))
    assert math.isclose(iou[0, 0], 50 / 150)


def test_perfect_detection_has_ap_one():
    assert detection_ap({"a": [(0.9, BOX)]}, {"a": [(BOX, False)]}) == 1.0


def test_missed_object_halves_ap():
    truth = {"a": [(BOX, False)], "b": [(BOX, False)]}
    assert detection_ap({"a": [(0.9, BOX)]}, truth) == 0.5


def test_false_positive_on_negative_image_ranked_first_lowers_ap():
    truth = {"a": [(BOX, False)], "negative": []}
    predictions = {"negative": [(0.9, BOX)], "a": [(0.5, BOX)]}
    assert detection_ap(predictions, truth) == 0.5


def test_duplicate_detection_is_a_false_positive():
    truth = {"a": [(BOX, False)]}
    assert detection_ap({"a": [(0.9, BOX), (0.8, BOX)]}, truth) == 1.0
    assert detection_ap({"a": [(0.9, FAR), (0.8, BOX)]}, truth) == 0.5


def test_group_of_counts_once_and_ignores_later_hits():
    truth = {"a": [([0.0, 0.0, 100.0, 100.0], True)]}
    inside = [10.0, 10.0, 20.0, 20.0]
    assert detection_ap({"a": [(0.9, inside), (0.8, inside)]}, truth) == 1.0


def test_unevaluated_image_predictions_are_ignored():
    assert detection_ap({"other": [(0.9, BOX)], "a": [(0.5, BOX)]}, {"a": [(BOX, False)]}) == 1.0


def test_mask_iou():
    a = np.array([[True, True], [False, False]])
    b = np.array([[True, False], [False, False]])
    assert mask_iou(a, b) == 0.5


def test_auroc_handles_single_class():
    assert auroc([1, 0], [0.9, 0.1]) == 1.0
    assert math.isnan(auroc([1, 1], [0.9, 0.1]))


def test_bootstrap_ci_contains_mean():
    values = np.arange(100.0)
    low, high = bootstrap_ci(lambda index: values[index].mean(), 100, samples=200)
    assert low < values.mean() < high


def test_box_labels_include_descendants_at_every_depth():
    hierarchy = {
        "LabelName": "root",
        "Subcategory": [
            {"LabelName": "/p", "Subcategory": [{"LabelName": "/m"}]},
            {"LabelName": "/b"},
        ],
    }
    names = {"/p": "Person", "/m": "Man", "/b": "Box"}
    assert expand_box_labels(hierarchy, names, ["Person", "Box"]) == {
        "Person": {"Person", "Man"},
        "Box": {"Box"},
    }
