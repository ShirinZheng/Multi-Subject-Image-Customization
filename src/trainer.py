"""Optional SDXL DreamBooth LoRA training utilities."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from .config import Config, SubjectConfig
from .data_quality import audit_reference_directory


def resolve_training_script() -> Path:
    """Resolve the official Diffusers SDXL DreamBooth LoRA script."""

    configured = os.getenv("DIFFUSERS_TRAIN_SCRIPT")
    candidates = [
        Path(configured) if configured else None,
        Config.PROJECT_ROOT.parent
        / "diffusers"
        / "examples"
        / "dreambooth"
        / "train_dreambooth_lora_sdxl.py",
        Path("/content/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "The official Diffusers training script was not found. Clone "
        "https://github.com/huggingface/diffusers and set DIFFUSERS_TRAIN_SCRIPT "
        "to examples/dreambooth/train_dreambooth_lora_sdxl.py."
    )


@contextmanager
def curated_reference_directory(subject: SubjectConfig) -> Iterator[Path]:
    """Expose one copy of each exact-duplicate group to the trainer."""

    audit = audit_reference_directory(subject.reference_dir)
    if len(audit.unique_files) < Config.MIN_UNIQUE_REFERENCE_IMAGES:
        raise ValueError(
            f"{subject.name} has only {len(audit.unique_files)} unique reference "
            f"image(s); at least {Config.MIN_UNIQUE_REFERENCE_IMAGES} are required."
        )

    with tempfile.TemporaryDirectory(prefix=f"{subject.name}-references-") as temp:
        destination = Path(temp)
        for index, source in enumerate(audit.unique_files, start=1):
            target = destination / f"{index:03d}{source.suffix.lower()}"
            shutil.copy2(source, target)
        yield destination


def build_training_command(
    subject: SubjectConfig,
    *,
    script: Path,
    instance_data_dir: Path,
) -> list[str]:
    if subject.lora_dir is None:
        raise ValueError(f"Subject {subject.name!r} does not define a LoRA output path.")

    return [
        "accelerate",
        "launch",
        str(script),
        f"--pretrained_model_name_or_path={Config.MODEL_NAME}",
        f"--instance_data_dir={instance_data_dir}",
        f"--output_dir={subject.lora_dir}",
        f"--instance_prompt={subject.instance_prompt}",
        "--resolution=1024",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--gradient_checkpointing",
        "--mixed_precision=fp16",
        "--use_8bit_adam",
        f"--learning_rate={Config.LEARNING_RATE}",
        f"--rank={Config.LORA_RANK}",
        f"--max_train_steps={Config.TRAIN_STEPS}",
        "--checkpointing_steps=400",
        "--checkpoints_total_limit=1",
        "--seed=42",
        f"--validation_prompt={subject.instance_prompt} in a natural indoor scene",
        "--num_validation_images=2",
        "--validation_epochs=50",
    ]


def run_training(
    subjects: Sequence[SubjectConfig] = Config.SUBJECTS,
    *,
    force: bool = False,
) -> None:
    """Train optional subject LoRAs without shell interpolation."""

    script = resolve_training_script()
    for subject in subjects:
        if subject.lora_dir is None:
            continue
        final_weights = subject.lora_dir / "pytorch_lora_weights.safetensors"
        if final_weights.exists() and not force:
            print(f"Skipping {subject.name}: final LoRA weights already exist.")
            continue

        subject.lora_dir.mkdir(parents=True, exist_ok=True)
        with curated_reference_directory(subject) as reference_dir:
            command = build_training_command(
                subject,
                script=script,
                instance_data_dir=reference_dir,
            )
            print(f"Training {subject.name} with curated unique references...")
            subprocess.run(command, check=True)
