"""Generate a vanilla SDXL baseline and the masked reference-conditioned result."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.multi_subject_pipeline import MultiSubjectPipeline


def generate_baseline():
    import torch
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
    )
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    generator = torch.Generator(device="cpu").manual_seed(Config.DEFAULT_SEED)
    return pipe(
        prompt=Config.DEFAULT_PROMPT,
        negative_prompt=Config.NEGATIVE_PROMPT,
        width=Config.WIDTH,
        height=Config.HEIGHT,
        num_inference_steps=Config.NUM_INFERENCE_STEPS,
        guidance_scale=Config.GUIDANCE_SCALE,
        generator=generator,
    ).images[0]


def run_experiment() -> None:
    Config.ensure_directories()
    print("Generating vanilla SDXL baseline...")
    baseline = generate_baseline()
    baseline.save(Config.OUTPUT_DIR / "baseline_result.png")

    print("Generating masked reference-conditioned candidates...")
    optimized = MultiSubjectPipeline().generate()
    optimized.image.save(Config.OUTPUT_DIR / "improved_result.png")
    optimized.save_report(Config.OUTPUT_DIR / "improved_result.json")
    print(f"Selected optimized score: {optimized.best_score:.4f}")


if __name__ == "__main__":
    run_experiment()
