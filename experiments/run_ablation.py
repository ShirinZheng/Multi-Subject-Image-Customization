"""Compare unmasked single-sample generation with the optimized pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.multi_subject_pipeline import MultiSubjectPipeline


def run_ablation() -> None:
    Config.ensure_directories()
    pipeline = MultiSubjectPipeline()

    print("Ablation A: one candidate without spatial masks")
    unmasked = pipeline.generate(
        seed=Config.DEFAULT_SEED,
        num_candidates=1,
        use_spatial_masks=False,
    )
    unmasked.image.save(Config.OUTPUT_DIR / "ablation_unmasked.png")
    unmasked.save_report(Config.OUTPUT_DIR / "ablation_unmasked.json")

    print("Ablation B: masked generation with candidate ranking")
    optimized = pipeline.generate(
        seed=Config.DEFAULT_SEED,
        num_candidates=Config.NUM_CANDIDATES,
        use_spatial_masks=True,
    )
    optimized.image.save(Config.OUTPUT_DIR / "ablation_optimized.png")
    optimized.save_report(Config.OUTPUT_DIR / "ablation_optimized.json")

    print(
        "Scores: "
        f"unmasked={unmasked.best_score:.4f}, "
        f"optimized={optimized.best_score:.4f}"
    )


if __name__ == "__main__":
    run_ablation()
