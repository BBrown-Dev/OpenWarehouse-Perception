import numpy as np
import pandas as pd
import pytest
from PIL import Image

from openwarehouse.robustness.corruptions import CORRUPTIONS, apply_corruption
from openwarehouse.robustness.runner import run_sweep, summarize_degradation


def photo():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_corruptions_keep_size_and_are_reproducible(name):
    image = photo()
    first = apply_corruption(image, name, 3, key="a")
    assert first.size == image.size and first.mode == "RGB"
    assert np.array_equal(np.asarray(first), np.asarray(apply_corruption(image, name, 3, key="a")))


def test_low_light_darkens_more_at_higher_severity():
    image = photo()
    means = [np.asarray(apply_corruption(image, "low_light", s, key="a")).mean() for s in (1, 3, 5)]
    assert means[0] > means[1] > means[2]


def test_occlusion_covers_at_least_the_target_area():
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    covered = (np.asarray(apply_corruption(image, "occlusion", 5, key="a"))[..., 0] > 100).mean()
    assert covered >= 0.4


def test_unknown_inputs_fail():
    with pytest.raises(KeyError):
        apply_corruption(photo(), "rain", 1)
    with pytest.raises(ValueError):
        apply_corruption(photo(), "glare", 6)


def test_sweep_includes_clean_and_reports_change():
    config = {"robustness": {"corruptions": ["dust_haze"], "severities": [1, 5]}}
    dataset = [("a", photo())]
    results = run_sweep(lambda images: [np.asarray(i).mean() for i in images], dataset, lambda p: {"brightness": p[0]}, config)
    assert list(results["corruption"]) == ["clean", "dust_haze", "dust_haze"]
    changes = summarize_degradation(results)["brightness_change"]
    assert changes.iloc[0] == 0 and isinstance(results, pd.DataFrame)


def test_sweep_cache_reuses_predictions_and_rejects_other_images(tmp_path):
    config = {"robustness": {"corruptions": ["dust_haze"], "severities": [1]}}
    calls = []

    def predict(images):
        calls.append(len(images))
        return [float(np.asarray(i).mean()) for i in images]

    first = run_sweep(predict, [("a", photo())], lambda p: {"v": p[0]}, config, cache_dir=tmp_path)
    second = run_sweep(predict, [("a", photo())], lambda p: {"v": p[0]}, config, cache_dir=tmp_path)
    assert len(calls) == 2 and first.equals(second)  # second sweep ran no predictions
    with pytest.raises(RuntimeError):
        run_sweep(predict, [("b", photo())], lambda p: {"v": p[0]}, config, cache_dir=tmp_path)
