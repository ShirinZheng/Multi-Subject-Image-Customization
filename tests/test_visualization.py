from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.config import Config
from src.data_quality import audit_reference_directory
from src.evaluation import SubjectScore
from src.multi_subject_pipeline import CandidateScore, GenerationResult
from src.visualization import Visualizer


def test_visualizer_saves_generation_grid(tmp_path: Path) -> None:
    scores = tuple(
        SubjectScore(
            subject=subject.name,
            identity_similarity=80,
            prompt_alignment=75,
            competing_similarity=10,
            leakage_penalty=0,
            composite=78.75,
        )
        for subject in Config.SUBJECTS
    )
    candidate = CandidateScore(seed=42, composite=78.75, subjects=scores)
    result = GenerationResult(
        image=Image.new("RGB", (128, 128), "gray"),
        prompt="test",
        best_seed=42,
        best_score=78.75,
        candidates=(candidate,),
        dataset_audits=tuple(
            audit_reference_directory(subject.reference_dir)
            for subject in Config.SUBJECTS
        ),
    )

    destination = Visualizer(tmp_path).save_result_grid(result)

    assert destination == tmp_path / "generation_report.png"
    assert destination.is_file()
    assert destination.stat().st_size > 0
