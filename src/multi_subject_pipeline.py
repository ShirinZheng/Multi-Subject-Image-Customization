"""Spatially isolated multi-subject image customization pipeline."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import Config, SubjectConfig
from .data_quality import DatasetAudit, validate_reference_sets
from .evaluation import Evaluator, SubjectScore
from .spatial_layout import LayoutGenerator


@dataclass(frozen=True)
class CandidateScore:
    seed: int
    composite: float
    subjects: tuple[SubjectScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "composite": self.composite,
            "subjects": [subject.to_dict() for subject in self.subjects],
        }


@dataclass
class GenerationResult:
    image: Image.Image
    prompt: str
    best_seed: int
    best_score: float
    candidates: tuple[CandidateScore, ...]
    dataset_audits: tuple[DatasetAudit, ...]

    def report(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "best_seed": self.best_seed,
            "best_score": self.best_score,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reference_data": [
                {
                    "directory": str(audit.directory),
                    "total_images": len(audit.files),
                    "unique_images": len(audit.unique_files),
                    "exact_duplicates": audit.duplicate_count,
                }
                for audit in self.dataset_audits
            ],
        }

    def save_report(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.report(), indent=2),
            encoding="utf-8",
        )


class MultiSubjectPipeline:
    """Generates and ranks masked IP-Adapter candidates.

    A single denoising pass composes all subjects, which keeps lighting and
    perspective coherent. Spatial IP-Adapter masks prevent one subject's image
    features from leaking into another subject's region.
    """

    def __init__(
        self,
        device: str | None = None,
        *,
        subjects: tuple[SubjectConfig, ...] = Config.SUBJECTS,
        pipe: Any | None = None,
        evaluator: Evaluator | None = None,
        mask_processor: Any | None = None,
        generator_factory: Callable[[int], Any] | None = None,
    ) -> None:
        self.device = device or self._default_device()
        self.subjects = tuple(subjects)
        self.layout = LayoutGenerator(
            width=Config.WIDTH,
            height=Config.HEIGHT,
            subjects=self.subjects,
        )
        self.evaluator = evaluator or Evaluator(device=self.device)
        self.pipe = pipe or self._build_pipeline()
        self._mask_processor = mask_processor
        self._generator_factory = generator_factory or self._make_generator
        self._adapter_loaded = pipe is not None
        self.last_result: GenerationResult | None = None

    @staticmethod
    def _default_device() -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build_pipeline(self) -> Any:
        try:
            import torch
            from diffusers import (
                AutoencoderKL,
                AutoPipelineForText2Image,
                EulerDiscreteScheduler,
            )
            from transformers import CLIPVisionModelWithProjection
        except ImportError as exc:
            raise RuntimeError(
                "Generation dependencies are not installed. "
                "Run `pip install -r requirements.txt` first."
            ) from exc

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            Config.IP_ADAPTER_REPO,
            subfolder=Config.IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER,
            torch_dtype=dtype,
        )
        vae = AutoencoderKL.from_pretrained(
            Config.VAE_NAME,
            torch_dtype=dtype,
        )
        pipe = AutoPipelineForText2Image.from_pretrained(
            Config.MODEL_NAME,
            image_encoder=image_encoder,
            vae=vae,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if self.device == "cuda" else None,
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_vae_slicing()

        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        return pipe

    def _make_generator(self, seed: int) -> Any:
        import torch

        return torch.Generator(device="cpu").manual_seed(seed)

    def _get_mask_processor(self) -> Any:
        if self._mask_processor is None:
            from diffusers.image_processor import IPAdapterMaskProcessor

            self._mask_processor = IPAdapterMaskProcessor()
        return self._mask_processor

    def load_reference_adapter(self) -> None:
        if self._adapter_loaded:
            return
        self.pipe.load_ip_adapter(
            Config.IP_ADAPTER_REPO,
            subfolder=Config.IP_ADAPTER_SUBFOLDER,
            weight_name=[Config.IP_ADAPTER_WEIGHT],
        )
        self._adapter_loaded = True

    def load_loras(self) -> None:
        """Load optional subject LoRAs for advanced experiments.

        The default generation path intentionally relies on masked reference
        conditioning. Loading multiple global LoRAs at once can reintroduce
        identity bleeding, so this method does not activate them automatically.
        """

        for subject in self.subjects:
            if subject.lora_dir is None:
                continue
            weight_path = subject.lora_dir / "pytorch_lora_weights.safetensors"
            if not weight_path.exists():
                continue
            self.pipe.load_lora_weights(
                subject.lora_dir,
                weight_name=weight_path.name,
                adapter_name=subject.name,
            )

    def _conditioning_inputs(
        self,
        audits: tuple[DatasetAudit, ...],
        *,
        use_spatial_masks: bool,
    ) -> tuple[list[list[Image.Image]], dict[str, Any]]:
        references: list[Image.Image] = []
        for audit in audits:
            with Image.open(audit.unique_files[0]) as image:
                references.append(image.convert("RGB").copy())

        self.pipe.set_ip_adapter_scale(
            [[subject.adapter_scale for subject in self.subjects]]
        )
        cross_attention_kwargs: dict[str, Any] = {}
        if use_spatial_masks:
            masks = self.layout.get_masks()
            processed = self._get_mask_processor().preprocess(
                masks,
                height=Config.HEIGHT,
                width=Config.WIDTH,
            )
            shape = processed.shape
            if len(shape) != 4:
                raise ValueError(
                    "IP-Adapter mask preprocessing must return a 4D tensor."
                )
            mask_batch = processed.reshape(1, shape[0], shape[2], shape[3])
            cross_attention_kwargs["ip_adapter_masks"] = [mask_batch]

        return [references], cross_attention_kwargs

    def _layout_prompt(self) -> str:
        positions: list[str] = []
        for subject in self.subjects:
            left, top, right, bottom = subject.region
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            horizontal = "left" if center_x < 0.4 else "right" if center_x > 0.6 else "center"
            vertical = "foreground" if center_y > 0.6 else "background"
            positions.append(
                f"{subject.description} appears once in the {horizontal} {vertical}"
            )
        return "; ".join(positions)

    def _evaluate_candidate(
        self,
        image: Image.Image,
        seed: int,
        audits: tuple[DatasetAudit, ...],
    ) -> CandidateScore:
        subject_scores: list[SubjectScore] = []
        for index, (subject, audit) in enumerate(
            zip(self.subjects, audits, strict=True)
        ):
            competing_paths = tuple(
                path
                for other_index, other_audit in enumerate(audits)
                if other_index != index
                for path in other_audit.unique_files
            )
            subject_scores.append(
                self.evaluator.evaluate_subject(
                    image=image,
                    subject_name=subject.name,
                    crop_box=self.layout.crop_box(subject.name),
                    prompt=subject.description,
                    reference_paths=audit.unique_files,
                    competing_reference_paths=competing_paths,
                )
            )
        composite = float(np.mean([score.composite for score in subject_scores]))
        return CandidateScore(
            seed=seed,
            composite=round(composite, 4),
            subjects=tuple(subject_scores),
        )

    def generate(
        self,
        prompt: str | None = None,
        *,
        seed: int = Config.DEFAULT_SEED,
        num_candidates: int = Config.NUM_CANDIDATES,
        use_spatial_masks: bool = True,
    ) -> GenerationResult:
        if num_candidates < 1:
            raise ValueError("num_candidates must be at least 1.")

        audits = validate_reference_sets(
            self.subjects,
            minimum_unique_images=Config.MIN_UNIQUE_REFERENCE_IMAGES,
        )
        self.load_reference_adapter()
        ip_adapter_images, cross_attention_kwargs = self._conditioning_inputs(
            audits,
            use_spatial_masks=use_spatial_masks,
        )

        user_prompt = (prompt or Config.DEFAULT_PROMPT).strip()
        composed_prompt = f"{user_prompt}. Layout constraints: {self._layout_prompt()}."
        images: list[Image.Image] = []
        scores: list[CandidateScore] = []

        for offset in range(num_candidates):
            candidate_seed = seed + offset
            output = self.pipe(
                prompt=composed_prompt,
                negative_prompt=Config.NEGATIVE_PROMPT,
                ip_adapter_image=ip_adapter_images,
                cross_attention_kwargs=cross_attention_kwargs or None,
                width=Config.WIDTH,
                height=Config.HEIGHT,
                num_inference_steps=Config.NUM_INFERENCE_STEPS,
                guidance_scale=Config.GUIDANCE_SCALE,
                generator=self._generator_factory(candidate_seed),
            )
            image = output.images[0].convert("RGB")
            images.append(image)
            scores.append(self._evaluate_candidate(image, candidate_seed, audits))

        best_index = max(range(len(scores)), key=lambda index: scores[index].composite)
        result = GenerationResult(
            image=images[best_index],
            prompt=composed_prompt,
            best_seed=scores[best_index].seed,
            best_score=scores[best_index].composite,
            candidates=tuple(scores),
            dataset_audits=audits,
        )
        self.last_result = result
        return result

    def generate_with_qa_loop(self, prompt: str | None = None) -> Image.Image:
        """Backward-compatible wrapper returning only the selected image."""

        return self.generate(prompt=prompt).image
