import gzip
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from openwarehouse.data.leakage import difference_hash, find_exclusions, hamming_distances
from openwarehouse.experiments import (
    _check_prediction_settings,
    freeze_selection,
    read_prediction_rows,
    require_selection,
    restrict_to_manifest,
)

MAX_DISTANCE = 24


def _photo(seed: int) -> Image.Image:
    generator = np.random.default_rng(seed)
    coarse = generator.integers(0, 255, (24, 24, 3), dtype=np.uint8)
    return Image.fromarray(coarse).resize((640, 480), Image.BICUBIC)


def _distance(first, second) -> int:
    return int(hamming_distances(first[None], second[None])[0, 0])


def test_hash_catches_recompressed_resized_copy_and_separates_different_photos(tmp_path):
    original, copy, other = (tmp_path / name for name in ("a.jpg", "b.jpg", "c.jpg"))
    _photo(1).save(original, quality=95)
    _photo(1).resize((320, 240)).save(copy, quality=40)
    _photo(2).save(other, quality=95)
    assert _distance(difference_hash(original), difference_hash(copy)) <= MAX_DISTANCE
    assert _distance(difference_hash(original), difference_hash(other)) > MAX_DISTANCE


def test_find_exclusions_flags_each_link_and_keeps_unrelated_images():
    images = pd.DataFrame(
        {
            "split": ["train", "test", "test", "test", "test"],
            "role": ["dev", "heldout", "heldout", "heldout", "heldout"],
            "image_id": ["d", "bytes", "near", "author", "clean"],
            "sha256": ["s", "s", "x1", "x2", "x3"],
            "author": ["me", "u1", "u2", "me", "u3"],
        }
    )
    base = np.zeros(256, dtype=bool)
    far = np.ones(256, dtype=bool)
    hashes = np.stack([base, far, base, far, far])
    found = find_exclusions(images, hashes, MAX_DISTANCE).set_index("image_id")
    assert set(found.index) == {"bytes", "near", "author"}
    assert found.loc["bytes", "same_bytes"]
    assert found.loc["near", "near_duplicate"]
    assert found.loc["author", "same_author"]


def test_heldout_is_refused_until_selection_is_frozen(tmp_path):
    with pytest.raises(RuntimeError):
        require_selection(tmp_path / "selection.json")


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["test"],
            "image_id": ["a"],
            "class_name": ["Box"],
            "is_positive": [True],
            "role": ["heldout"],
        }
    )


def test_restrict_drops_rows_outside_manifest_and_takes_labels_from_it():
    rows = pd.DataFrame(
        {
            "split": ["test", "test"],
            "image_id": ["a", "excluded"],
            "class_name": ["Box", "Box"],
            "condition": ["bare", "bare"],
            "is_positive": [False, True],
        }
    )
    restricted = restrict_to_manifest(rows, _manifest())
    assert restricted["image_id"].tolist() == ["a"]
    assert restricted["is_positive"].tolist() == [True]


def test_restrict_fails_when_a_manifest_row_has_no_result():
    rows = pd.DataFrame(
        {"split": ["test"], "image_id": ["other"], "class_name": ["Box"], "condition": ["bare"]}
    )
    with pytest.raises(ValueError):
        restrict_to_manifest(rows, _manifest())


def test_read_prediction_rows_recovers_rows_from_truncated_file(tmp_path):
    path = tmp_path / "predictions.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for index in range(2000):
            handle.write(json.dumps({"index": index, "padding": str(index) * 20}) + "\n")
    path.write_bytes(path.read_bytes()[:-2000])
    rows, is_complete = read_prediction_rows(path)
    assert not is_complete
    assert rows
    assert [row["index"] for row in rows] == list(range(len(rows)))


def test_read_prediction_rows_reports_complete_file(tmp_path):
    path = tmp_path / "predictions.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"index": 0}) + "\n")
    assert read_prediction_rows(path) == ([{"index": 0}], True)


def test_selection_cannot_be_frozen_twice(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError):
        freeze_selection(pd.DataFrame(), pd.DataFrame(), 0.5, path)


def test_prediction_cache_rejects_changed_settings(tmp_path):
    path = tmp_path / "meta.json"
    config = {"models": {"detector": "a"}, "inference": {"threshold": 0.1}}
    _check_prediction_settings(config, path)
    _check_prediction_settings(config, path)
    with pytest.raises(RuntimeError):
        _check_prediction_settings({**config, "inference": {"threshold": 0.2}}, path)
