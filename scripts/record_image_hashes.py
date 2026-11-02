"""Record, or verify, the SHA-256 of every image behind the results.

Image IDs alone do not pin the data: a re-download can return different bytes or fail. Run with
`--verify` before reproducing results to confirm the local images match the recorded snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from openwarehouse.config import load_config
from openwarehouse.experiments import OUTPUT_DIR, image_path
from openwarehouse.gate import FRESH_MANIFEST_PATH

SNAPSHOT = Path("docs/results/image_sha256.csv")


def current(config) -> pd.DataFrame:
    frames = [pd.read_csv(OUTPUT_DIR / "manifest.csv", dtype={"image_id": str}).assign(source="manifest")]
    if FRESH_MANIFEST_PATH.exists():
        frames.append(pd.read_csv(FRESH_MANIFEST_PATH, dtype={"image_id": str}).assign(source="gate_fresh"))
    images = pd.concat(frames)[["source", "split", "image_id"]].drop_duplicates(["split", "image_id"])
    hashes = []
    for split, image_id in zip(images["split"], images["image_id"]):
        path = image_path(config, split, image_id)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing")
    return images.assign(sha256=hashes).sort_values(["split", "image_id"], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    found = current(load_config("open_images", "poc"))
    if not args.verify:
        found.to_csv(SNAPSHOT, index=False)
        print(found.groupby("source").size().to_dict(), "missing:", int((found["sha256"] == "missing").sum()))
        return
    recorded = pd.read_csv(SNAPSHOT, dtype={"image_id": str})
    merged = recorded.merge(found, on=["split", "image_id"], how="outer", suffixes=("_recorded", "_local"))
    bad = merged[merged["sha256_recorded"] != merged["sha256_local"]]
    print(f"{len(merged) - len(bad)} of {len(merged)} images match the snapshot")
    if len(bad):
        print(bad.head(20).to_string())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
