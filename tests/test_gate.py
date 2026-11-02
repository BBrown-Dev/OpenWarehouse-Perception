import numpy as np
import pytest

from openwarehouse.gate import freeze_gate, image_rates, recall_floor_cut, require_gate


def test_recall_floor_keeps_at_least_the_requested_share():
    positive = np.array([True] * 100 + [False])
    logits = np.array([*range(100), 500.0])
    cut = recall_floor_cut(positive, logits, 0.95)
    assert (logits[positive] >= cut).mean() >= 0.95
    assert cut == 5.0


def test_image_rates():
    y = np.array([True, True, False, False])
    fired = np.array([True, False, True, False])
    assert image_rates(y, fired) == {"precision": 0.5, "recall": 0.5, "fpr": 0.5}


def test_gate_cannot_be_frozen_twice_and_is_required(tmp_path):
    path = tmp_path / "gate.json"
    with pytest.raises(RuntimeError):
        require_gate(path)
    freeze_gate({"Person": 1.0}, path)
    assert require_gate(path) == {"Person": 1.0}
    with pytest.raises(RuntimeError):
        freeze_gate({"Person": 2.0}, path)
