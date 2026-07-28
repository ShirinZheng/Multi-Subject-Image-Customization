"""Command-line demo for masked multi-subject customization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.multi_subject_pipeline import MultiSubjectPipeline
from src.visualization import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a spatially isolated multi-subject image."
    )
    parser.add_argument(
        "--prompt",
        default=Config.DEFAULT_PROMPT,
        help="Scene and interaction prompt.",
    )
    parser.add_argument("--seed", type=int, default=Config.DEFAULT_SEED)
    parser.add_argument(
        "--candidates",
        type=int,
        default=Config.NUM_CANDIDATES,
        help="Number of candidates to generate and rank.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Config.OUTPUT_DIR / "improved_result.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Config.ensure_directories()

    pipeline = MultiSubjectPipeline(device=args.device)
    result = pipeline.generate(
        prompt=args.prompt,
        seed=args.seed,
        num_candidates=args.candidates,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(args.output)
    report_path = args.output.with_suffix(".json")
    result.save_report(report_path)
    grid_path = Visualizer(args.output.parent).save_result_grid(result)

    print(f"Selected seed: {result.best_seed}")
    print(f"Composite score: {result.best_score:.4f}")
    print(f"Image: {args.output}")
    print(f"Metrics: {report_path}")
    print(f"Visual report: {grid_path}")


if __name__ == "__main__":
    main()
