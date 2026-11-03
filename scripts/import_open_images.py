"""Select warehouse images from Open Images V7 and import them through FiftyOne.

By default the committed manifest (`docs/results/manifest.csv`) is reused, so a new machine
downloads exactly the images the results were computed on. Sampling again (`--resample`) checks
which development images the public bucket still holds, so it can select a different set and
should only be used to build a new evaluation.
"""

from __future__ import annotations

import argparse

import pandas as pd

from openwarehouse.config import PROJECT_ROOT, load_config
from openwarehouse.data.open_images import (
    fetch_labels,
    import_with_fiftyone,
    select_manifest,
    summarize_manifest,
)

COMMITTED_MANIFEST = PROJECT_ROOT / "docs" / "results" / "manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resample", action="store_true", help="draw a new sample instead of the committed manifest")
    args = parser.parse_args()
    config = load_config("open_images")
    if COMMITTED_MANIFEST.exists() and not args.resample:
        manifest = pd.read_csv(COMMITTED_MANIFEST, dtype={"image_id": str})
        print(f"using the committed manifest: {COMMITTED_MANIFEST}")
    else:
        labels = fetch_labels(config)
        manifest = select_manifest(labels, config, seed=config["seed"])
        print("drew a new sample; results will not match the committed ones")
    output_dir = PROJECT_ROOT / "artifacts" / "poc"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    print(summarize_manifest(manifest).to_string())
    print("images to import:", manifest[["image_id", "split"]].drop_duplicates().shape[0])
    dataset = import_with_fiftyone(manifest, config)
    print(dataset)


if __name__ == "__main__":
    main()
