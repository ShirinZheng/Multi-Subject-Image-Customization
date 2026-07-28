# Architecture

## Goal

The pipeline receives multiple reference subjects and a scene prompt, then
generates one coherent 1024 × 1024 image. It must preserve each identity,
respect the requested spatial relationship, and avoid duplicated or blended
subjects.

## Design Decisions

### One shared denoising process

The legacy pipeline generated a full scene and then replaced large regions one
at a time. Each inpainting pass could damage previous subjects and introduce
lighting discontinuities.

The optimized pipeline conditions one SDXL denoising process on all subjects.
This gives the model one global lighting, depth, and perspective solution.

### Spatially masked image conditioning

IP-Adapter separates image and text cross-attention. A binary mask associates
each reference image with one output region. The layout validator rejects
overlapping regions before inference.

The masks control feature placement, while the text prompt controls the shared
scene and interaction. Neither input is expected to solve both tasks alone.

### Reference-aware candidate selection

The pipeline generates several deterministic seeds. Each candidate is cropped
according to the configured regions and compared with:

1. the intended subject references;
2. the intended subject description;
3. the competing subjects' references.

The final score rewards identity and prompt alignment while penalizing
cross-subject leakage. The highest-scoring complete image is selected; no
post-selection patchwork is performed.

## Runtime Sequence

1. Resolve portable project paths.
2. Audit every reference directory and group exact duplicates.
3. Validate normalized, non-overlapping layout regions.
4. Load SDXL, CLIP ViT-H, and IP-Adapter Plus.
5. Build one conditioning image and mask per subject.
6. Generate `N` deterministic candidates.
7. Load the lightweight evaluation CLIP model on demand.
8. Score every subject crop in every candidate.
9. Select the best complete image.
10. Save the image, JSON metrics, and visual report.

## Why LoRA Is Optional

Subject LoRAs can provide useful detail when trained on a diverse dataset, but
multiple global LoRAs are not spatially isolated. They can bind one subject's
attributes to another region and recreate the identity-bleeding problem.

The default path therefore uses reference conditioning. LoRA training remains
available for controlled experiments and future localized refinement work.

## Evaluation Boundaries

CLIP is a ranking signal. It does not prove exact identity, anatomical
correctness, factual accuracy, or safety. For people, brand assets, small text,
or high-risk applications, add a domain-specific identity metric and human
review.
