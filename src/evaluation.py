"""Region-aware image quality evaluation.

The original project measured the full output against a text prompt. That score
cannot tell whether a particular subject matches its reference image. This
module evaluates each subject crop against its own references and against the
other subjects' references to expose identity bleeding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .config import Config


class EmbeddingBackend(Protocol):
    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


class HuggingFaceClipBackend:
    """Lazily loads CLIP so importing the project does not allocate GPU memory."""

    def __init__(
        self,
        model_name: str = Config.CLIP_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, CLIPModel

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self.device = device
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(device)
        self._model.eval()

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.clip(norms, 1e-12, None)

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        self._load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        inputs = self._processor(images=list(images), return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            features = self._model.get_image_features(**inputs)
        return self._normalize(features.detach().float().cpu().numpy())

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        inputs = self._processor(
            text=list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            features = self._model.get_text_features(**inputs)
        return self._normalize(features.detach().float().cpu().numpy())


@dataclass(frozen=True)
class SubjectScore:
    subject: str
    identity_similarity: float
    prompt_alignment: float
    competing_similarity: float
    leakage_penalty: float
    composite: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


class Evaluator:
    """Computes reference-aware scores for subject-specific image regions."""

    def __init__(
        self,
        backend: EmbeddingBackend | None = None,
        device: str | None = None,
    ) -> None:
        self.backend = backend or HuggingFaceClipBackend(device=device)
        self._reference_cache: dict[tuple[str, ...], np.ndarray] = {}

    @staticmethod
    def _cosine_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        query = query / np.clip(np.linalg.norm(query, axis=1, keepdims=True), 1e-12, None)
        candidates = candidates / np.clip(
            np.linalg.norm(candidates, axis=1, keepdims=True),
            1e-12,
            None,
        )
        return query @ candidates.T

    def _reference_embeddings(self, paths: Sequence[Path | str]) -> np.ndarray:
        key = tuple(str(Path(path).resolve()) for path in paths)
        if key not in self._reference_cache:
            images: list[Image.Image] = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            self._reference_cache[key] = self.backend.encode_images(images)
        return self._reference_cache[key]

    def evaluate_subject(
        self,
        *,
        image: Image.Image,
        subject_name: str,
        crop_box: tuple[int, int, int, int],
        prompt: str,
        reference_paths: Sequence[Path | str],
        competing_reference_paths: Sequence[Path | str] = (),
    ) -> SubjectScore:
        if not reference_paths:
            raise ValueError(f"Subject {subject_name!r} has no reference images.")

        crop = image.convert("RGB").crop(crop_box)
        crop_embedding = self.backend.encode_images([crop])
        reference_embeddings = self._reference_embeddings(reference_paths)
        identity = float(self._cosine_scores(crop_embedding, reference_embeddings).max() * 100)

        text_embedding = self.backend.encode_texts([prompt])
        prompt_alignment = float(self._cosine_scores(crop_embedding, text_embedding)[0, 0] * 100)

        competing_similarity = 0.0
        if competing_reference_paths:
            competing_embeddings = self._reference_embeddings(competing_reference_paths)
            competing_similarity = float(
                self._cosine_scores(crop_embedding, competing_embeddings).max() * 100
            )

        # A five-point margin prevents harmless class-level similarity from
        # being mistaken for identity leakage.
        leakage_penalty = max(0.0, competing_similarity - identity + 5.0)
        composite = 0.75 * identity + 0.25 * prompt_alignment - 0.5 * leakage_penalty

        return SubjectScore(
            subject=subject_name,
            identity_similarity=round(identity, 4),
            prompt_alignment=round(prompt_alignment, 4),
            competing_similarity=round(competing_similarity, 4),
            leakage_penalty=round(leakage_penalty, 4),
            composite=round(composite, 4),
        )

    def compute_clip_score(self, image: Image.Image, prompt: str) -> float:
        """Backward-compatible full-image text alignment score."""

        image_embedding = self.backend.encode_images([image.convert("RGB")])
        text_embedding = self.backend.encode_texts([prompt])
        return float(self._cosine_scores(image_embedding, text_embedding)[0, 0] * 100)
