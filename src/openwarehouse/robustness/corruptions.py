"""Warehouse-relevant image corruptions, registered by the names used in the config.

Sizes are relative to the image, so a severity means the same thing at any resolution. Every
random choice is seeded from the caller's key, so a corrupted image is reproducible.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageFilter

SEVERITIES = (1, 2, 3, 4, 5)


def _rng(key: str, name: str, severity: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{key}|{name}|{severity}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _image(pixels: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(pixels, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def low_light(image, severity, rng):
    """Underexposure with the sensor noise that comes with it."""
    scale = (0.6, 0.45, 0.3, 0.2, 0.12)[severity - 1]
    noise = (0.01, 0.015, 0.02, 0.03, 0.04)[severity - 1]
    pixels = _array(image) ** 1.2 * scale
    return _image(pixels + rng.normal(0, noise, pixels.shape))


def glare(image, severity, rng):
    """A bright spot, as from a light source or reflective floor."""
    pixels = _array(image)
    height, width = pixels.shape[:2]
    cy, cx = rng.uniform(0.2, 0.8) * height, rng.uniform(0.2, 0.8) * width
    radius = (0.15, 0.2, 0.25, 0.32, 0.4)[severity - 1] * max(height, width)
    strength = (0.35, 0.5, 0.65, 0.8, 0.95)[severity - 1]
    y, x = np.ogrid[:height, :width]
    falloff = np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / (2 * (radius / 2) ** 2)))[..., None]
    return _image(pixels + strength * falloff)


def motion_blur(image, severity, rng):
    """Horizontal motion blur, as from a moving forklift or camera."""
    pixels = _array(image)
    length = max(3, int((0.005, 0.01, 0.02, 0.03, 0.045)[severity - 1] * pixels.shape[1]))
    kernel = np.ones(length, dtype=np.float32) / length
    blurred = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, pixels)
    return _image(blurred)


def defocus_blur(image, severity, rng):
    radius = (0.002, 0.004, 0.007, 0.011, 0.016)[severity - 1] * max(image.size)
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius))


def gaussian_noise(image, severity, rng):
    pixels = _array(image)
    sigma = (0.04, 0.08, 0.12, 0.18, 0.26)[severity - 1]
    return _image(pixels + rng.normal(0, sigma, pixels.shape))


def jpeg_compression(image, severity, rng):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=(50, 30, 15, 8, 4)[severity - 1])
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def dust_haze(image, severity, rng):
    """Airborne dust or a dirty lens: contrast lost toward a light gray."""
    alpha = (0.15, 0.25, 0.38, 0.5, 0.65)[severity - 1]
    return _image(_array(image) * (1 - alpha) + 0.75 * alpha)


def occlusion(image, severity, rng):
    """Gray rectangles, as from racking or loads, at random positions over the scene."""
    pixels = _array(image).copy()
    height, width = pixels.shape[:2]
    target = (0.05, 0.1, 0.18, 0.28, 0.4)[severity - 1] * height * width
    covered = np.zeros((height, width), dtype=bool)
    while covered.sum() < target:
        h = int(rng.uniform(0.1, 0.35) * height)
        w = int(rng.uniform(0.1, 0.35) * width)
        y, x = int(rng.uniform(0, height - h)), int(rng.uniform(0, width - w))
        covered[y : y + h, x : x + w] = True
    pixels[covered] = 0.5
    return _image(pixels)


CORRUPTIONS: dict[str, Callable[[Image.Image, int, np.random.Generator], Image.Image]] = {
    "low_light": low_light,
    "glare": glare,
    "motion_blur": motion_blur,
    "defocus_blur": defocus_blur,
    "gaussian_noise": gaussian_noise,
    "jpeg_compression": jpeg_compression,
    "dust_haze": dust_haze,
    "occlusion": occlusion,
}


def apply_corruption(image: Any, name: str, severity: int, key: str = "") -> Any:
    """Apply one configured corruption at severity 1 to 5, reproducibly for a given key."""
    if name not in CORRUPTIONS:
        raise KeyError(f"Unknown corruption: {name}")
    if severity not in SEVERITIES:
        raise ValueError("Severity must be 1 to 5.")
    return CORRUPTIONS[name](image, severity, _rng(key, name, severity))
