"""Visual reports for layouts, outputs, and subject-level detail crops."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .config import Config
from .multi_subject_pipeline import GenerationResult
from .spatial_layout import LayoutGenerator


class Visualizer:
    def __init__(self, save_dir: Path | str = Config.OUTPUT_DIR) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_result_grid(
        self,
        result: GenerationResult,
        *,
        filename: str = "generation_report.png",
    ) -> Path:
        layout = LayoutGenerator(
            width=result.image.width,
            height=result.image.height,
        )
        columns = 2 + len(layout.subjects)
        figure, axes = plt.subplots(1, columns, figsize=(6 * columns, 6))

        axes[0].imshow(layout.preview())
        axes[0].set_title("Spatial conditioning regions")
        axes[1].imshow(result.image)
        axes[1].set_title(
            f"Selected candidate\nseed={result.best_seed}, score={result.best_score:.2f}"
        )

        best_candidate = next(
            candidate
            for candidate in result.candidates
            if candidate.seed == result.best_seed
        )
        for index, (subject, score) in enumerate(
            zip(layout.subjects, best_candidate.subjects, strict=True),
            start=2,
        ):
            axes[index].imshow(
                result.image.crop(layout.crop_box(subject.name))
            )
            axes[index].set_title(
                f"{subject.name}\n"
                f"identity={score.identity_similarity:.2f}, "
                f"composite={score.composite:.2f}"
            )

        for axis in axes:
            axis.axis("off")

        destination = self.save_dir / filename
        figure.tight_layout()
        figure.savefig(destination, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return destination
