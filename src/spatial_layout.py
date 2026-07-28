"""Layout regions and masks for spatially isolated subject conditioning."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

from PIL import Image, ImageDraw, ImageFilter

from .config import Config, SubjectConfig


@dataclass(frozen=True)
class LayoutRegion:
    name: str
    normalized_box: tuple[float, float, float, float]

    def pixel_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        left, top, right, bottom = self.normalized_box
        return (
            round(left * width),
            round(top * height),
            round(right * width),
            round(bottom * height),
        )


class LayoutGenerator:
    """Builds non-overlapping masks from normalized subject regions."""

    def __init__(
        self,
        width: int = Config.WIDTH,
        height: int = Config.HEIGHT,
        subjects: Iterable[SubjectConfig] = Config.SUBJECTS,
    ) -> None:
        self.width = width
        self.height = height
        self.subjects = tuple(subjects)
        self.regions = {
            subject.name: LayoutRegion(subject.name, subject.region)
            for subject in self.subjects
        }
        self._validate()

    def _validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Layout width and height must be positive.")

        for region in self.regions.values():
            left, top, right, bottom = region.normalized_box
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise ValueError(
                    f"Region {region.name!r} must stay inside normalized bounds."
                )

        for first, second in combinations(self.regions.values(), 2):
            if self._intersection_area(
                first.normalized_box,
                second.normalized_box,
            ) > 0:
                raise ValueError(
                    f"Regions {first.name!r} and {second.name!r} overlap. "
                    "Overlapping subject masks increase identity bleeding."
                )

    @staticmethod
    def _intersection_area(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        return max(0.0, right - left) * max(0.0, bottom - top)

    def crop_box(self, subject_name: str, padding: float = 0.02) -> tuple[int, int, int, int]:
        left, top, right, bottom = self.regions[subject_name].normalized_box
        expanded = (
            max(0.0, left - padding),
            max(0.0, top - padding),
            min(1.0, right + padding),
            min(1.0, bottom + padding),
        )
        return LayoutRegion(subject_name, expanded).pixel_box(self.width, self.height)

    def mask_for(
        self,
        subject_name: str,
        *,
        feather_radius: int = 0,
    ) -> Image.Image:
        mask = Image.new("L", (self.width, self.height), 0)
        box = self.regions[subject_name].pixel_box(self.width, self.height)
        radius = max(8, round(min(box[2] - box[0], box[3] - box[1]) * 0.08))
        ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius, fill=255)
        if feather_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        return mask

    def get_masks(self, *, feather_radius: int = 0) -> list[Image.Image]:
        return [
            self.mask_for(subject.name, feather_radius=feather_radius)
            for subject in self.subjects
        ]

    def get_dual_subject_layout(self) -> tuple[Image.Image, list[Image.Image]]:
        """Compatibility wrapper for the original public API."""

        base_image = Image.new("RGB", (self.width, self.height), "white")
        return base_image, self.get_masks()

    def preview(self) -> Image.Image:
        colors = (
            (238, 92, 83),
            (70, 130, 180),
            (116, 188, 98),
            (179, 126, 209),
        )
        preview = Image.new("RGB", (self.width, self.height), (25, 25, 28))
        for index, subject in enumerate(self.subjects):
            color = Image.new("RGB", preview.size, colors[index % len(colors)])
            preview = Image.composite(color, preview, self.mask_for(subject.name))
        return preview
