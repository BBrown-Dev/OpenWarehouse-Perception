"""SigLIP text-image scoring."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModel, AutoProcessor


def _as_embedding(output: Any) -> torch.Tensor:
    """Return the embedding tensor, whichever output type this transformers version uses."""
    return output if isinstance(output, torch.Tensor) else output.pooler_output


class VisionLanguageEncoder:
    """Scores image-text pairs independently with SigLIP's sigmoid head.

    Sigmoid scores suit an open vocabulary: unlike softmax, a score does not change when the
    other candidate labels change, and every label can be absent from an image.
    """

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    @torch.inference_mode()
    def encode_images(self, images: list[Any]) -> torch.Tensor:
        """Return L2-normalized image embeddings."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.autocast(self.device.split(":")[0], enabled=self.device != "cpu"):
            embeddings = _as_embedding(self.model.get_image_features(**inputs))
        return torch.nn.functional.normalize(embeddings.float(), dim=-1)

    @torch.inference_mode()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        """Return L2-normalized text embeddings.

        SigLIP was trained on text padded to 64 tokens, so max-length padding is required.
        """
        inputs = self.processor(
            text=prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
        ).to(self.device)
        embeddings = _as_embedding(self.model.get_text_features(**inputs))
        return torch.nn.functional.normalize(embeddings.float(), dim=-1)

    def logits(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply the learned temperature and bias, giving one logit per image-text pair."""
        scale = self.model.logit_scale.exp()
        return image_embeddings @ text_embeddings.T * scale + self.model.logit_bias

    def score(self, images: list[Any], prompts: list[str]) -> torch.Tensor:
        """Return an images x prompts matrix of match probabilities."""
        logits = self.logits(self.encode_images(images), self.encode_text(prompts))
        return torch.sigmoid(logits)
