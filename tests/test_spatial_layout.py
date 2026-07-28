from __future__ import annotations

import numpy as np
import pytest

from src.config import PROJECT_ROOT, SubjectConfig
from src.spatial_layout import LayoutGenerator


def make_subject(name: str, region: tuple[float, float, float, float]) -> SubjectConfig:
    return SubjectConfig(
        name=name,
        class_name=name,
        description=name,
        reference_dir=PROJECT_ROOT / "data" / name,
        region=region,
    )


def test_masks_do_not_overlap() -> None:
    subjects = (
        make_subject("left", (0.05, 0.1, 0.45, 0.9)),
        make_subject("right", (0.55, 0.1, 0.95, 0.9)),
    )
    layout = LayoutGenerator(width=128, height=128, subjects=subjects)
    left, right = [np.asarray(mask) > 0 for mask in layout.get_masks()]

    assert not np.logical_and(left, right).any()
    assert layout.crop_box("left")[0] == 4
    assert layout.crop_box("right")[2] == 124


def test_overlapping_regions_are_rejected() -> None:
    subjects = (
        make_subject("first", (0.1, 0.1, 0.6, 0.9)),
        make_subject("second", (0.5, 0.1, 0.9, 0.9)),
    )

    with pytest.raises(ValueError, match="identity bleeding"):
        LayoutGenerator(subjects=subjects)
