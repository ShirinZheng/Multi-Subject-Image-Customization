"""Project configuration.

The defaults are intentionally portable. Every path is derived from the
repository root instead of assuming a Google Colab mount point.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


@dataclass(frozen=True)
class SubjectConfig:
    """A customized subject and its intended output region."""

    name: str
    class_name: str
    description: str
    reference_dir: Path
    region: tuple[float, float, float, float]
    adapter_scale: float = 0.78
    trigger_token: str | None = None
    lora_dir: Path | None = None

    @property
    def instance_prompt(self) -> str:
        prefix = f"{self.trigger_token} " if self.trigger_token else ""
        return f"a photo of {prefix}{self.class_name}"


class Config:
    """Runtime defaults shared by training, generation, and experiments."""

    PROJECT_ROOT = PROJECT_ROOT
    OUTPUT_DIR = PROJECT_ROOT / "output"
    CACHE_DIR = PROJECT_ROOT / ".cache"

    MODEL_NAME = os.getenv(
        "MULTISUBJECT_MODEL",
        "stabilityai/stable-diffusion-xl-base-1.0",
    )
    IP_ADAPTER_REPO = os.getenv("MULTISUBJECT_IP_ADAPTER_REPO", "h94/IP-Adapter")
    IP_ADAPTER_SUBFOLDER = "sdxl_models"
    IP_ADAPTER_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"
    IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER = "models/image_encoder"
    VAE_NAME = os.getenv(
        "MULTISUBJECT_VAE",
        "madebyollin/sdxl-vae-fp16-fix",
    )
    CLIP_MODEL_NAME = os.getenv(
        "MULTISUBJECT_CLIP_MODEL",
        "openai/clip-vit-base-patch32",
    )

    WIDTH = _env_int("MULTISUBJECT_WIDTH", 1024)
    HEIGHT = _env_int("MULTISUBJECT_HEIGHT", 1024)
    NUM_INFERENCE_STEPS = _env_int("MULTISUBJECT_STEPS", 35)
    GUIDANCE_SCALE = _env_float("MULTISUBJECT_GUIDANCE", 6.5)
    NUM_CANDIDATES = _env_int("MULTISUBJECT_CANDIDATES", 3)
    DEFAULT_SEED = _env_int("MULTISUBJECT_SEED", 42)

    # Optional DreamBooth LoRA training. The reference-conditioned pipeline does
    # not require these checkpoints, but users can retrain them for experiments.
    TRAIN_STEPS = _env_int("MULTISUBJECT_TRAIN_STEPS", 800)
    LEARNING_RATE = _env_float("MULTISUBJECT_LEARNING_RATE", 1e-4)
    LORA_RANK = _env_int("MULTISUBJECT_LORA_RANK", 16)
    MIN_UNIQUE_REFERENCE_IMAGES = 2
    RECOMMENDED_UNIQUE_REFERENCE_IMAGES = 8

    DEFAULT_PROMPT = (
        "A photorealistic gray tabby cat with a white chest and white paws sits "
        "on the left side of a continuous dark walnut table beside a small "
        "glossy red ceramic cup on the right, warm natural window light, "
        "consistent perspective, realistic contact shadows, shallow depth of field"
    )
    NEGATIVE_PROMPT = (
        "duplicate subject, extra cat, extra cup, merged subjects, fused anatomy, "
        "identity bleeding, ghost object, transparent object, split screen, collage, "
        "disconnected background, malformed face, malformed paws, deformed handle, "
        "cropped subject, blurry subject, low resolution, text, watermark, logo"
    )

    SUBJECTS = (
        SubjectConfig(
            name="tabby_cat",
            class_name="gray tabby cat",
            description="the same gray tabby cat with a white chest and white paws",
            reference_dir=PROJECT_ROOT / "data" / "cat_toy",
            region=(0.07, 0.18, 0.57, 0.91),
            adapter_scale=0.82,
            trigger_token="sks",
            lora_dir=PROJECT_ROOT / "checkpoints" / "lora_cat",
        ),
        SubjectConfig(
            name="red_mug",
            class_name="small red ceramic cup",
            description="the same small glossy red ceramic cup with one rounded handle",
            reference_dir=PROJECT_ROOT / "data" / "red_mug",
            region=(0.63, 0.43, 0.94, 0.83),
            adapter_scale=0.74,
            trigger_token="trk",
            lora_dir=PROJECT_ROOT / "checkpoints" / "lora_mug",
        ),
    )

    @classmethod
    def ensure_directories(cls) -> None:
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
