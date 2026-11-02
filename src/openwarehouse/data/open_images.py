"""Open Images V7 selection and import through FiftyOne.

FiftyOne's `classes` filter only accepts the 600 boxable classes, which would silently drop
`Forklift truck`, `Warehouse`, and `Pallet`. Images are therefore selected here from the official
human-verified image-level labels, then passed to FiftyOne as `image_ids`.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import requests
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

from openwarehouse.config import PROJECT_ROOT

OI_FILE_PREFIX = {"train": "train", "validation": "val", "test": "test"}


def _annotations_dir(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / config["open_images"]["annotations_dir"]


def _all_classes(config: dict[str, Any]) -> list[str]:
    return config["boxable_classes"] + config["image_level_classes"] + config["dev_only_classes"]


def load_class_ids(config: dict[str, Any]) -> dict[str, str]:
    """Return a map of class name to Open Images MID, failing on any unknown name."""
    path = _annotations_dir(config) / "oidv7-class-descriptions.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{config['open_images']['base_url']}/oidv7-class-descriptions.csv"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    with path.open(encoding="utf-8") as handle:
        name_to_mid = {row[1]: row[0] for row in csv.reader(handle) if len(row) == 2}
    missing = [name for name in _all_classes(config) if name not in name_to_mid]
    if missing:
        raise ValueError(f"Classes not in Open Images V7: {missing}")
    return {name: name_to_mid[name] for name in _all_classes(config)}


def fetch_labels(config: dict[str, Any]) -> pd.DataFrame:
    """Stream the verified label files, keeping only rows for the configured classes."""
    class_ids = load_class_ids(config)
    mid_to_name = {mid: name for name, mid in class_ids.items()}
    cache = _annotations_dir(config) / "selected_labels.csv"
    if cache.exists():
        labels = pd.read_csv(cache)
        if set(labels["class_name"]) == set(class_ids):
            return labels

    rows = []
    for split, prefix in OI_FILE_PREFIX.items():
        url = (
            f"{config['open_images']['base_url']}/"
            f"oidv7-{prefix}-annotations-human-imagelabels.csv"
        )
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            for raw in response.iter_lines():
                # CRLF line endings can yield an empty line at a chunk boundary.
                if not raw.strip():
                    continue
                image_id, _, mid, confidence = raw.decode("utf-8").split(",")
                if mid in mid_to_name:
                    rows.append((image_id, split, mid_to_name[mid], float(confidence) >= 0.5))
    labels = pd.DataFrame(rows, columns=["image_id", "split", "class_name", "is_positive"])
    labels = labels.drop_duplicates(["image_id", "split", "class_name"])
    labels.to_csv(cache, index=False)
    return labels


def expand_box_labels(
    hierarchy: dict[str, Any], mid_to_name: dict[str, str], class_names: list[str]
) -> dict[str, set[str]]:
    """Map each class to itself and its descendants in the box label hierarchy.

    Open Images labels boxes at the most specific level, so most people are `Man` or `Woman`, not
    `Person`. Its evaluation counts those boxes for the parent class, and so must this project.
    """
    expanded = {name: {name} for name in class_names}

    def walk(node: dict[str, Any], ancestors: list[str]) -> None:
        name = mid_to_name.get(node["LabelName"], node["LabelName"])
        for ancestor in ancestors:
            if ancestor in expanded:
                expanded[ancestor].add(name)
        for child in node.get("Subcategory", []):
            walk(child, [*ancestors, name])

    walk(hierarchy, [])
    return expanded


def load_box_labels(config: dict[str, Any]) -> dict[str, set[str]]:
    """Return each boxable class's box labels, from the official files beside the dataset."""
    metadata = PROJECT_ROOT / config["open_images"]["dataset_dir"] / "test" / "metadata"
    hierarchy = json.loads((metadata / "hierarchy.json").read_text(encoding="utf-8"))
    with (metadata / "classes.csv").open(encoding="utf-8") as handle:
        mid_to_name = {row[0]: row[1] for row in csv.reader(handle) if len(row) == 2}
    return expand_box_labels(hierarchy, mid_to_name, config["boxable_classes"])


def _s3_client() -> Any:
    return boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=16))


def _sample_available(subset: pd.DataFrame, cap: int, bucket: str, seed: int) -> pd.DataFrame:
    """Sample rows whose images exist in the bucket.

    The train labels cover far more images than the public bucket holds, so a label alone does
    not mean an image can be downloaded.
    """
    client = _s3_client()

    def exists(row: tuple[str, str]) -> bool:
        split, image_id = row
        try:
            client.head_object(Bucket=bucket, Key=f"{split}/{image_id}.jpg")
            return True
        except ClientError:
            return False

    shuffled = subset.sample(frac=1, random_state=seed)
    kept = []
    for start in range(0, len(shuffled), cap):
        chunk = shuffled.iloc[start : start + cap]
        with ThreadPoolExecutor(max_workers=16) as pool:
            flags = list(pool.map(exists, zip(chunk["split"], chunk["image_id"])))
        kept.append(chunk[flags])
        if sum(len(part) for part in kept) >= cap:
            break
    return pd.concat(kept).head(cap)


def select_manifest(labels: pd.DataFrame, config: dict[str, Any], seed: int) -> pd.DataFrame:
    """Sample capped positives and negatives for each class in each evaluation role.

    A sampled image keeps every verified label it has, so negatives stay honest for all classes.
    """
    settings = config["open_images"]
    labels = labels.assign(role=labels["split"].map(settings["role_by_split"]))
    selected = []
    for (role, class_name), group in labels.groupby(["role", "class_name"]):
        if role == "heldout" and class_name in config["dev_only_classes"]:
            continue
        caps = settings["sample_caps"][role]
        for is_positive, subset in group.groupby("is_positive"):
            cap = caps["positive" if is_positive else "negative"]
            if role == "dev":
                selected.append(_sample_available(subset, cap, settings["image_bucket"], seed))
            else:
                selected.append(subset.sample(min(cap, len(subset)), random_state=seed))
    chosen = pd.concat(selected)[["image_id", "split"]].drop_duplicates()
    return labels.merge(chosen, on=["image_id", "split"]).sort_values(
        ["role", "split", "image_id", "class_name"], ignore_index=True
    )


def import_with_fiftyone(manifest: pd.DataFrame, config: dict[str, Any]) -> Any:
    """Download the selected images and their labels into one persistent FiftyOne dataset."""
    import fiftyone as fo
    import fiftyone.zoo as foz

    settings = config["open_images"]
    name = settings["fiftyone_dataset"]
    # The zoo appends the dataset name to this directory, giving `dataset_dir`.
    fo.config.dataset_zoo_dir = str((PROJECT_ROOT / settings["dataset_dir"]).parent)
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    dataset = fo.Dataset(name, persistent=True)
    for split, group in manifest.groupby("split"):
        view = foz.load_zoo_dataset(
            settings["zoo_name"],
            split=split,
            image_ids=sorted(group["image_id"].unique()),
            label_types=settings["label_types_by_split"][split],
            dataset_name=f"{name}-{split}",
            drop_existing_dataset=True,
            shuffle=False,
        )
        # With a single label type, FiftyOne names the field `ground_truth` instead.
        if "ground_truth" in view.get_field_schema():
            view.rename_sample_field("ground_truth", "detections")
        dataset.add_samples(view)
        fo.delete_dataset(f"{name}-{split}")

        # The loader skips images with no labels of the requested types. Those are often the
        # image-level classes this project needs, so they are downloaded directly.
        data_dir = PROJECT_ROOT / settings["dataset_dir"] / split / "data"
        image_ids = sorted(group["image_id"].unique())
        added = {Path(path).stem for path in dataset.values("filepath")}
        missing = [image_id for image_id in image_ids if image_id not in added]
        print(f"{split}: adding {len(missing)} images without requested labels")
        paths = _download_images(missing, split, data_dir, settings["image_bucket"])
        dataset.add_samples([fo.Sample(filepath=str(path)) for path in paths])

    expected = manifest[["split", "image_id"]].drop_duplicates().shape[0]
    print(f"imported {dataset.count('filepath')} of {expected} selected images")
    return dataset


def _download_images(image_ids: list[str], split: str, data_dir: Path, bucket: str) -> list[Path]:
    """Download images from the public bucket, skipping any no longer available."""
    client = _s3_client()
    data_dir.mkdir(parents=True, exist_ok=True)

    def fetch(image_id: str) -> Path | None:
        path = data_dir / f"{image_id}.jpg"
        if not path.exists():
            try:
                client.download_file(bucket, f"{split}/{image_id}.jpg", str(path))
            except ClientError as error:
                print(f"could not download {split}/{image_id}: {error}")
                return None
        return path

    with ThreadPoolExecutor(max_workers=16) as pool:
        return [path for path in pool.map(fetch, image_ids) if path is not None]


def summarize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Count verified positives and negatives for each class and role."""
    return (
        manifest.groupby(["class_name", "role", "is_positive"])["image_id"]
        .nunique()
        .unstack(["role", "is_positive"], fill_value=0)
    )
