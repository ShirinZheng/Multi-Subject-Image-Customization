from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation import Evaluator


class ColorBackend:
    def encode_images(self, images):
        vectors = []
        for image in images:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
            vectors.append(pixels.mean(axis=(0, 1)))
        return np.asarray(vectors)

    def encode_texts(self, texts):
        vectors = []
        for text in texts:
            if "red" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "blue" in text:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([1.0, 1.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def test_subject_score_rewards_target_and_rejects_competitor(tmp_path: Path) -> None:
    target = tmp_path / "red.png"
    competitor = tmp_path / "blue.png"
    Image.new("RGB", (16, 16), "red").save(target)
    Image.new("RGB", (16, 16), "blue").save(competitor)
    candidate = Image.new("RGB", (32, 32), "red")

    score = Evaluator(backend=ColorBackend()).evaluate_subject(
        image=candidate,
        subject_name="red_subject",
        crop_box=(0, 0, 32, 32),
        prompt="the same red object",
        reference_paths=[target],
        competing_reference_paths=[competitor],
    )

    assert score.identity_similarity == 100
    assert score.prompt_alignment == 100
    assert score.competing_similarity == 0
    assert score.leakage_penalty == 0
    assert score.composite == 100


def test_subject_score_penalizes_competing_identity(tmp_path: Path) -> None:
    target = tmp_path / "blue.png"
    competitor = tmp_path / "red.png"
    Image.new("RGB", (16, 16), "blue").save(target)
    Image.new("RGB", (16, 16), "red").save(competitor)
    candidate = Image.new("RGB", (32, 32), "red")
    evaluator = Evaluator(backend=ColorBackend())

    score = evaluator.evaluate_subject(
        image=candidate,
        subject_name="blue_subject",
        crop_box=(0, 0, 32, 32),
        prompt="the same blue object",
        reference_paths=[target],
        competing_reference_paths=[competitor],
    )

    assert score.competing_similarity == 100
    assert score.leakage_penalty == 105
    assert score.composite < 0
    assert score.to_dict()["subject"] == "blue_subject"
    assert evaluator.compute_clip_score(candidate, "a red object") == 100


def test_subject_score_requires_references() -> None:
    evaluator = Evaluator(backend=ColorBackend())

    with np.testing.assert_raises_regex(ValueError, "no reference images"):
        evaluator.evaluate_subject(
            image=Image.new("RGB", (16, 16), "red"),
            subject_name="missing",
            crop_box=(0, 0, 16, 16),
            prompt="red",
            reference_paths=[],
        )
