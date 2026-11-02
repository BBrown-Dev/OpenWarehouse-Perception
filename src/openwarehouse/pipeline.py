"""End-to-end inference: natural-language prompt to scored, segmented instances."""

from __future__ import annotations

from typing import Any

from openwarehouse.models.vision_language import VisionLanguageEncoder
from openwarehouse.schemas import Instance
from openwarehouse.segmentation.grounded_sam import GroundedSAM, rescore_instances


class PerceptionPipeline:
    """Grounding DINO proposes boxes, SAM segments them, and SigLIP checks each masked crop."""

    def __init__(self, config: dict[str, Any], device: str = "cuda") -> None:
        self.grounded_sam = GroundedSAM(config, device=device)
        self.encoder = VisionLanguageEncoder(config["models"]["siglip"], device=device)

    def __call__(
        self, image: Any, prompt: str, min_score: float = 0.0, encoder_prompt: str | None = None
    ) -> list[Instance]:
        """Return instances for one prompt. An empty list means nothing passed the threshold.

        `encoder_prompt` lets SigLIP use its own wording. The two models prefer different prompts.
        """
        detections = self.grounded_sam.detect(image.convert("RGB"), prompt)
        kept = [instance for instance in detections if instance.score >= min_score]
        instances = self.grounded_sam.segment(image.convert("RGB"), kept)
        return rescore_instances(
            image.convert("RGB"), instances, self.encoder, encoder_prompt or prompt
        )
