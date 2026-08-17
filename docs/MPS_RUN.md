# Apple Silicon (MPS) Run Guide

This guide records a successful end-to-end SDXL and masked IP-Adapter run on an
Apple M3 with 16 GB unified memory. It is intended as a reproducible quality
validation, not a lightweight smoke test.

## Validated Environment

| Item | Value |
| --- | --- |
| Hardware | Apple M3, 16 GB unified memory |
| Operating system | macOS 14.5 |
| Python | 3.10.11 |
| PyTorch device | `mps` |
| Output size | 768 × 768 |
| Inference steps | 28 |
| Candidates | 3 (Seeds 42, 43, and 44) |
| Selected candidate | Seed 42 |
| Composite ranking score | 73.0284 |

The score is a relative CLIP-based ranking signal. The selected image and
subject crops were also inspected visually.

## Setup

Create the environment and install the pinned compatible dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first generation downloads SDXL, IP-Adapter Plus, its vision encoder, the
FP16-safe VAE, and CLIP. Keep at least 20 GB of free disk space for model caches
and temporary files.

Verify that PyTorch can use Metal:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

The command should print `True`.

## Quality Run

Run from the repository root after activating the virtual environment:

```bash
HF_HOME="$HOME/.cache/huggingface" \
HF_HUB_DISABLE_XET=1 \
HF_HUB_DOWNLOAD_TIMEOUT=300 \
HF_HUB_ETAG_TIMEOUT=30 \
TOKENIZERS_PARALLELISM=false \
MULTISUBJECT_WIDTH=768 \
MULTISUBJECT_HEIGHT=768 \
MULTISUBJECT_STEPS=28 \
MULTISUBJECT_CANDIDATES=3 \
python examples/run_demo.py \
  --device mps \
  --candidates 3 \
  --prompt "Photorealistic gray tabby cat with white chest and paws on the left, small glossy red ceramic mug with one handle on the right, both on one walnut table, warm window light, natural shadows" \
  --output output/high_quality_768.png
```

The output contains the selected image, a JSON report for every candidate, and
a visual report with the configured spatial regions and subject crops.

## Result

![Selected Seed 42 result](../output/high_quality_768.png)

The three ranked candidates scored:

| Seed | Composite score |
| --- | ---: |
| 42 | 73.0284 |
| 43 | 72.1358 |
| 44 | 69.7488 |

The selected subject scores were:

| Subject | Identity similarity | Prompt alignment | Composite |
| --- | ---: | ---: | ---: |
| Tabby cat | 91.1264 | 32.2097 | 76.3972 |
| Red mug | 83.8045 | 27.2255 | 69.6597 |

See the [JSON metrics](../output/high_quality_768.json) and
[visual report](../output/generation_report.png) for full details.

## MPS-Specific Behavior

- SDXL, the VAE, and IP-Adapter image encoder use FP16 on MPS to stay within the
  unified-memory budget.
- The CLIP evaluator stays on CPU on MPS systems. Loading it beside SDXL on the
  GPU caused later candidates to slow down because both models competed for
  unified memory.
- The automatic layout suffix is kept short so both subject constraints remain
  inside CLIP's 77-token input limit.
- `transformers` is pinned below major version 5 for compatibility with the
  supported Diffusers release range.

If MPS reports an out-of-memory error, close other GPU-heavy applications or
reduce the run to 640 × 640, 20 steps, and two candidates. A 512 × 512,
four-step run is useful only for checking that the pipeline starts; it is not a
meaningful image-quality configuration.
