"""Grounded SAM: Grounding DINO boxes used as SAM prompts."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

from openwarehouse.schemas import Instance


def to_caption(prompt: str) -> str:
    """Format a prompt as Grounding DINO expects: lowercase and ending in a period."""
    return prompt.strip().lower().rstrip(".") + "."


class GroundedSAM:
    def __init__(self, config: dict[str, Any], device: str = "cuda") -> None:
        self.device = device
        self.use_autocast = device != "cpu"
        self.settings = config["inference"]
        models = config["models"]
        self.detector_processor = AutoProcessor.from_pretrained(models["detector"])
        self.detector_processor.image_processor.size = {
            "shortest_edge": self.settings["detector_shortest_edge"],
            "longest_edge": self.settings["detector_longest_edge"],
        }
        self.detector = (
            AutoModelForZeroShotObjectDetection.from_pretrained(models["detector"])
            .to(device)
            .eval()
        )
        self.segmenter_id = models["segmenter"]
        self._segmenter: SamModel | None = None
        self._segmenter_processor: SamProcessor | None = None

    def _run_detector(self, images: list[Any], captions: list[str]) -> tuple[Any, Any]:
        inputs = self.detector_processor(
            images=images, text=captions, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.autocast(self.device.split(":")[0], enabled=self.use_autocast):
            outputs = self.detector(**inputs)
        outputs.logits = outputs.logits.float()
        outputs.pred_boxes = outputs.pred_boxes.float()
        return inputs, outputs

    @torch.inference_mode()
    def detect_batch(self, images: list[Any], prompt: str) -> list[list[Instance]]:
        """Return text-prompted detections for each image, sharing one prompt."""
        caption = to_caption(prompt)
        inputs, outputs = self._run_detector(images, [caption] * len(images))
        results = self.detector_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.settings["detector_threshold"],
            text_threshold=self.settings["text_threshold"],
            target_sizes=[image.size[::-1] for image in images],
        )
        batches = []
        for result in results:
            order = result["scores"].argsort(descending=True)[: self.settings["max_detections"]]
            batches.append(
                [
                    Instance(
                        label=result["text_labels"][i],
                        score=float(result["scores"][i]),
                        box_xyxy=tuple(float(v) for v in result["boxes"][i]),
                    )
                    for i in order.tolist()
                ]
            )
        return batches

    def detect(self, image: Any, prompt: str) -> list[Instance]:
        """Return detections with boxes and no masks."""
        return self.detect_batch([image], prompt)[0]

    @torch.inference_mode()
    def token_activations(self, image: Any, prompt: str, top_k: int = 1) -> list[dict[str, Any]]:
        """Return each token's activation for the highest-scoring queries.

        Shows which prompt words the detection is actually aligned to.
        """
        inputs, outputs = self._run_detector([image], [to_caption(prompt)])
        probabilities = outputs.logits.sigmoid()[0]
        tokens = self.detector_processor.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        top_queries = probabilities.max(dim=-1).values.topk(top_k).indices
        return [
            {
                "score": float(probabilities[query].max()),
                "tokens": {
                    f"{index}:{token}": round(float(probabilities[query, index]), 4)
                    for index, token in enumerate(tokens)
                },
            }
            for query in top_queries.tolist()
        ]

    def _load_segmenter(self) -> None:
        if self._segmenter is None:
            self._segmenter_processor = SamProcessor.from_pretrained(self.segmenter_id)
            self._segmenter = SamModel.from_pretrained(self.segmenter_id).to(self.device).eval()

    @torch.inference_mode()
    def segment(self, image: Any, instances: list[Instance]) -> list[Instance]:
        """Add a SAM mask to each detection, using its box as the only prompt."""
        if not instances:
            return instances
        self._load_segmenter()
        inputs = self._segmenter_processor(
            images=image,
            input_boxes=[[list(instance.box_xyxy) for instance in instances]],
            return_tensors="pt",
        ).to(self.device)
        outputs = self._segmenter(**inputs, multimask_output=False)
        masks = self._segmenter_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        for instance, mask in zip(instances, masks):
            instance.mask = mask[0].numpy().astype(bool)
        return instances

    def predict(self, image: Any, prompt: str) -> list[Instance]:
        """Run detection and then segmentation."""
        return self.segment(image, self.detect(image, prompt))


def masked_crop(image: Any, instance: Instance) -> Any:
    """Crop a box and gray out everything outside its mask."""
    box = tuple(int(round(v)) for v in instance.box_xyxy)
    pixels = np.array(image.convert("RGB"))
    if instance.mask is not None:
        pixels[~instance.mask] = 127
    crop = Image.fromarray(pixels)
    width, height = image.size
    x0, y0 = max(box[0], 0), max(box[1], 0)
    x1, y1 = min(max(box[2], x0 + 1), width), min(max(box[3], y0 + 1), height)
    return crop.crop((x0, y0, x1, y1))


def rescore_instances(
    image: Any,
    instances: list[Instance],
    encoder: Any,
    prompt: str,
) -> list[Instance]:
    """Set `encoder_score` on each instance by scoring its masked crop against the prompt."""
    if not instances:
        return instances
    crops = [masked_crop(image, instance) for instance in instances]
    scores = encoder.score(crops, [prompt])[:, 0]
    for instance, score in zip(instances, scores.tolist()):
        instance.encoder_score = score
    return instances
