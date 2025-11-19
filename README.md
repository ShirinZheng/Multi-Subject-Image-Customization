# Multi-Subject Image Customization
A novel Coarse-to-Fine framework with Active QA Feedback for generating high-fidelity images containing multiple customized subjects while ensuring seamless spatial composition and identity preservation.

## Project Overview
This project addresses the critical challenge of "Identity Bleeding" and "Spatial Inconsistency" in multi-subject text-to-image generation. Unlike standard baselines that struggle to separate concepts, our approach utilizes a Layout-Aware Inpainting Pipeline combined with an Automatic Quality Assurance (QA) Loop.

### Key Innovations
1. **Coarse-to-Fine Composition Pipeline**: A two-stage generation process that first establishes a coherent global scene (Coarse) using base SDXL, followed by localized Identity Injection (Fine) using subject-specific LoRAs. This ensures perfect lighting consistency and spatial logic.

2. **Active QA & Self-Correction Agent**: A closed-loop feedback system that evaluates generated subjects in real-time using CLIP scores. If a subject's identity fidelity falls below a threshold, the system automatically triggers a Targeted Refinement Pass to restore features without altering the global composition.

### Results
- **Identity Preservation**: High CLIP-I scores through dedicated LoRA adapters.

- **Spatial Coherence**: Zero "split-world" artifacts due to global scene initialization.

- **Robustness**: Automatic error correction via the QA Agent eliminates "ghost objects" or "faded identities".

### Requirements

### Hardware

- GPU: NVIDIA L4 / A100 (24GB+ VRAM recommended for SDXL).

- Environment: Optimized for Google Colab.

### Software
- Python 3.10+

- PyTorch 2.0+

- Diffusers (Latest)

- Peft (for LoRA)

## Installation
### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/MultiSubjectGen_Pro.git
cd MultiSubjectGen_Pro
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
#Note: The project relies on madebyollin/sdxl-vae-fp16-fix to prevent VAE numerical instability in FP16 mode.
```

## Usage
## Quick Start

Run the full end-to-end demo. This script handles training check, baseline comparison, ablation studies, and final visual report generation.

```bash
python examples/run_demo.py
```
Configuration
Configure your subjects and training parameters in src/config.py:

```bash
SUBJECTS = [
    {
        "name": "cat_toy",
        "token": "sks",  # Trigger word for Subject A
        "class": "cat",
        "data": "data/cat_toy",
        "lora_out": "checkpoints/lora_cat"
    },
    # ... Add more subjects here
]
```

# QA Threshold Settings
QA_PASS_THRESHOLD = 23.0  # Score to trigger refinement
Training Custom Subjects
The system automatically checks for existing checkpoints. To force retrain:

```bash
from src.trainer import run_training

# Trains LoRAs for all subjects defined in config.py
run_training()
```
# Running Experiments
Baseline Comparison
Compare the naive SDXL generation against our Coarse-to-Fine approach:

```bash
python experiments/run_baseline_comparison.py
```
Output: Generates baseline_result.png vs ours_result.png in the output/ directory.

# Ablation Study
Verify the effectiveness of the Active QA Loop:

```bash
python experiments/run_ablation.py
```
Output: Generates comparison images showing results with and without the QA self-correction module.

# Evaluation
Automated Metrics
The pipeline includes an embedded evaluator that reports CLIP Scores for Identity Preservation and Prompt Adherence during generation.

```bash
from src.evaluation import Evaluator

evaluator = Evaluator()
score = evaluator.compute_clip_score(image, "a photo of sks cat")
print(f"Identity Fidelity Score: {score}")
```

## Project Structure

```
Multi-Subject-Image-Customization/
├── src/
│   ├── config.py                 # Global configuration & Subject definitions
│   ├── multi_subject_pipeline.py # Core: Coarse-to-Fine logic + QA Loop
│   ├── trainer.py                # SDXL LoRA training wrapper
│   ├── spatial_layout.py         # Layout mask generator
│   ├── evaluation.py             # CLIP evaluation metrics
│   └── visualization.py          # Report visualization tools
├── experiments/
│   ├── run_baseline_comparison.py
│   └── run_ablation.py           # QA Loop ablation study
├── examples/
│   └── run_demo.py               # Main entry point
├── data/                         # Training images (User Uploaded)
├── output/                       # Generated results
└── requirements.txt                   
└── README.md
```

## Technical Details
Architecture
Base Model: stabilityai/stable-diffusion-xl-base-1.0

## Pipeline Stages:

Phase 0 (Global): Generate a coherent background/scene using vanilla SDXL (No LoRA) to ensure unified lighting and perspective.

Phase 1 (Injection): Inpaint subjects into specific layout zones using dedicated LoRAs.

Phase 2 (QA & Refinement): Evaluate Identity Score. If low, trigger a high-strength refinement pass (strength=0.65) with boosted prompt weights.

## Optimization:

FP16 Pipeline: Full half-precision inference for speed.

VAE Fix: Uses sdxl-vae-fp16-fix to avoid black-image artifacts common in SDXL FP16.

## Troubleshooting
VAE "Black Image" or NaN Errors
We use a specific fixed VAE (madebyollin/sdxl-vae-fp16-fix) to resolve standard SDXL numerical overflow issues in FP16. Ensure your internet connection allows downloading from HuggingFace.

## Identity Bleeding (Subjects mixing)
If subjects mix (e.g., "cat looks like a mug"), ensure your layout masks in src/spatial_layout.py have sufficient separation, or increase the QA_PASS_THRESHOLD in config.py to force stricter refinement.

## Citation
If you use this code in your research, please cite:

```bibtex
@article{multisubject2025,
  title={Multi-Subject Image Customization via Coarse-to-Fine Active QA},
  author={[Your Name]},
  journal={Computer Vision Course Final Project},
  year={2025}
}
```

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Email: [shirinzheng871@gmail.com]
