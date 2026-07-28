from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.config import Config, SubjectConfig
from src.evaluation import SubjectScore
from src.multi_subject_pipeline import MultiSubjectPipeline


class FakeMaskProcessor:
    def preprocess(self, masks, *, height, width):
        assert len(masks) == 2
        return np.zeros((2, 1, height, width), dtype=np.uint8)


class FakePipe:
    def __init__(self):
        self.scales = None
        self.calls = []
        self.reference_adapter_loads = []
        self.lora_loads = []

    def set_ip_adapter_scale(self, scales):
        self.scales = scales

    def load_ip_adapter(self, *args, **kwargs):
        self.reference_adapter_loads.append((args, kwargs))

    def load_lora_weights(self, *args, **kwargs):
        self.lora_loads.append((args, kwargs))

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        seed = kwargs["generator"]
        image = Image.new("RGB", (Config.WIDTH, Config.HEIGHT), (seed, 0, 0))
        return SimpleNamespace(images=[image])


class FakeEvaluator:
    def evaluate_subject(self, *, image, subject_name, **kwargs):
        del kwargs
        value = float(image.getpixel((0, 0))[0])
        return SubjectScore(
            subject=subject_name,
            identity_similarity=value,
            prompt_alignment=value,
            competing_similarity=0,
            leakage_penalty=0,
            composite=value,
        )


def create_subject(
    tmp_path: Path,
    name: str,
    region: tuple[float, float, float, float],
    color: str,
) -> SubjectConfig:
    references = tmp_path / name
    references.mkdir()
    Image.new("RGB", (16, 16), color).save(references / "01.png")
    Image.new("RGB", (16, 16), color).resize((17, 17)).save(references / "02.png")
    lora_dir = tmp_path / f"{name}-lora"
    lora_dir.mkdir()
    (lora_dir / "pytorch_lora_weights.safetensors").write_bytes(b"weights")
    return SubjectConfig(
        name=name,
        class_name=name,
        description=f"the same {name}",
        reference_dir=references,
        region=region,
        adapter_scale=0.75,
        lora_dir=lora_dir,
    )


def build_pipeline(tmp_path: Path) -> tuple[MultiSubjectPipeline, FakePipe]:
    subjects = (
        create_subject(tmp_path, "red", (0.05, 0.1, 0.45, 0.9), "red"),
        create_subject(tmp_path, "blue", (0.55, 0.1, 0.95, 0.9), "blue"),
    )
    fake_pipe = FakePipe()
    pipeline = MultiSubjectPipeline(
        device="cpu",
        subjects=subjects,
        pipe=fake_pipe,
        evaluator=FakeEvaluator(),
        mask_processor=FakeMaskProcessor(),
        generator_factory=lambda seed: seed,
    )
    return pipeline, fake_pipe


def test_generate_selects_best_candidate_and_writes_report(tmp_path: Path) -> None:
    pipeline, fake_pipe = build_pipeline(tmp_path)

    result = pipeline.generate(
        prompt="red and blue subjects share one scene",
        seed=10,
        num_candidates=3,
    )

    assert result.best_seed == 12
    assert result.best_score == 12
    assert len(result.candidates) == 3
    assert fake_pipe.scales == [[0.75, 0.75]]
    assert len(fake_pipe.calls) == 3
    assert fake_pipe.calls[0]["cross_attention_kwargs"][
        "ip_adapter_masks"
    ][0].shape == (1, 2, Config.HEIGHT, Config.WIDTH)
    assert "Layout constraints" in fake_pipe.calls[0]["prompt"]

    report_path = tmp_path / "result.json"
    result.save_report(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["best_seed"] == 12
    assert report["reference_data"][0]["unique_images"] == 2


def test_pipeline_validates_candidate_count_and_loads_optional_adapters(
    tmp_path: Path,
) -> None:
    pipeline, fake_pipe = build_pipeline(tmp_path)

    with pytest.raises(ValueError, match="at least 1"):
        pipeline.generate(num_candidates=0)

    pipeline._adapter_loaded = False
    pipeline.load_reference_adapter()
    pipeline.load_reference_adapter()
    assert len(fake_pipe.reference_adapter_loads) == 1

    pipeline.load_loras()
    assert len(fake_pipe.lora_loads) == 2


def test_unmasked_generation_omits_attention_masks(tmp_path: Path) -> None:
    pipeline, fake_pipe = build_pipeline(tmp_path)

    image = pipeline.generate(
        seed=20,
        num_candidates=1,
        use_spatial_masks=False,
    ).image

    assert image.getpixel((0, 0))[0] == 20
    assert fake_pipe.calls[0]["cross_attention_kwargs"] is None


def test_build_pipeline_has_clear_missing_dependency_error() -> None:
    pipeline = MultiSubjectPipeline.__new__(MultiSubjectPipeline)
    pipeline.device = "cpu"

    with pytest.raises(RuntimeError, match="requirements.txt"):
        pipeline._build_pipeline()
