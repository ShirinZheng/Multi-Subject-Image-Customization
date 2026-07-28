# Reference Data Guide

Reference quality is the strongest controllable input to subject identity.

## Recommended Dataset

Provide 8 to 15 unique images for each subject:

- 2 to 4 front views;
- 2 to 4 three-quarter views;
- 1 to 3 side views;
- a mix of close and medium framing;
- several backgrounds and lighting conditions.

Keep the subject's defining features visible. For an animal, this may include
face markings, chest color, paws, ears, and collar. For an object, it may
include shape, handle placement, surface finish, logo, and proportions.

## Avoid

- exact or near-duplicate files;
- screenshots with app controls or watermarks;
- multiple target subjects in one reference;
- extreme filters or color casts;
- strong motion blur;
- heavy occlusion;
- images where the subject occupies a very small part of the frame;
- inconsistent identity or different versions of the object.

## Current Example Dataset

The repository's cat directory originally contained five files but only two
unique images. Exact duplicates do not add new identity information and can
overweight one pose during training.

The optimized runtime automatically ignores exact duplicates. For a meaningful
LoRA retraining experiment, replace the duplicates with at least six additional
unique views.

## Adding New Subjects

1. Create one directory per subject under `data/`.
2. Add the unique reference images.
3. Add a `SubjectConfig` entry in `src/config.py`.
4. Assign a normalized region that does not overlap existing regions.
5. Start with an adapter scale between `0.70` and `0.82`.
6. Run the unit tests before GPU inference.
7. Generate at least three candidates and inspect every subject crop.

## Privacy

Only use images you are authorized to process. Model inference may download
artifacts from Hugging Face, and hosted GPU environments may retain files or
logs. Review the provider's privacy, retention, and regional processing terms
before using personal or confidential images.
