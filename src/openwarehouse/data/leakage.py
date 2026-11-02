"""Checks that development data cannot inform held-out results.

Open Images splits have disjoint IDs, but its photos come from Flickr, so the same or a nearly
identical photo, or a photo from the same photographer's shoot, can appear in two splits.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from openwarehouse.config import PROJECT_ROOT

HASH_SIZE = 16


def difference_hash(path: Path) -> np.ndarray:
    """Return a perceptual hash as a bit array. It survives resizing and recompression."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((HASH_SIZE + 1, HASH_SIZE)), dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def hamming_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return pairwise differing bit counts between two boolean hash matrices."""
    first, second = first.astype(np.int32), second.astype(np.int32)
    return first.sum(1, keepdims=True) + second.sum(1) - 2 * first @ second.T


def fingerprint_images(images: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    """Return `images` with exact hashes added, and a matrix of perceptual hashes."""
    root = PROJECT_ROOT / config["open_images"]["dataset_dir"]

    def fingerprint(row: tuple[str, str]) -> tuple[str, np.ndarray]:
        path = root / row[0] / "data" / f"{row[1]}.jpg"
        return hashlib.sha256(path.read_bytes()).hexdigest(), difference_hash(path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fingerprint, zip(images["split"], images["image_id"])))
    return images.assign(sha256=[r[0] for r in results]), np.stack([r[1] for r in results])


def load_authors(images: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add each image's Flickr profile URL, which unlike a display name identifies one person."""
    root = PROJECT_ROOT / config["open_images"]["dataset_dir"]
    frames = []
    for split, group in images.groupby("split"):
        wanted = set(group["image_id"])
        path = root / split / "metadata" / "image_ids.csv"
        columns = ["ImageID", "AuthorProfileURL"]
        for chunk in pd.read_csv(path, dtype=str, usecols=columns, chunksize=500_000):
            frames.append(chunk[chunk["ImageID"].isin(wanted)])
    authors = pd.concat(frames).drop_duplicates("ImageID")
    authors.columns = ["image_id", "author"]
    return images.merge(authors, on="image_id", how="left")


def find_exclusions(
    images: pd.DataFrame, hashes: np.ndarray, max_distance: int
) -> pd.DataFrame:
    """Return held-out images linked to dev by ID, exact copy, near-duplicate, or photographer.

    `images` needs `role`, `image_id`, `sha256`, and `author` columns, with rows matching `hashes`.
    """
    is_dev = (images["role"] == "dev").to_numpy()
    dev, heldout = images[is_dev], images[~is_dev]
    distances = hamming_distances(hashes[~is_dev], hashes[is_dev])
    reasons = pd.DataFrame(
        {
            "split": heldout["split"].to_numpy(),
            "image_id": heldout["image_id"].to_numpy(),
            "same_id": heldout["image_id"].isin(dev["image_id"]).to_numpy(),
            "same_bytes": heldout["sha256"].isin(dev["sha256"]).to_numpy(),
            "near_duplicate": distances.min(axis=1) <= max_distance,
            "same_author": heldout["author"].isin(dev["author"].dropna()).to_numpy(),
        }
    )
    return reasons[reasons.drop(columns=["split", "image_id"]).any(axis=1)].reset_index(drop=True)
