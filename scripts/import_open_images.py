"""Select warehouse images from Open Images V7 and import them through FiftyOne."""

from __future__ import annotations

from openwarehouse.config import PROJECT_ROOT, load_config
from openwarehouse.data.open_images import (
    fetch_labels,
    import_with_fiftyone,
    select_manifest,
    summarize_manifest,
)


def main() -> None:
    config = load_config("open_images")
    labels = fetch_labels(config)
    manifest = select_manifest(labels, config, seed=config["seed"])
    output_dir = PROJECT_ROOT / "artifacts" / "poc"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    print(summarize_manifest(manifest).to_string())
    print("images to import:", manifest[["image_id", "split"]].drop_duplicates().shape[0])
    dataset = import_with_fiftyone(manifest, config)
    print(dataset)


if __name__ == "__main__":
    main()
