"""Reference-image validation and exact duplicate detection."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class DatasetAudit:
    directory: Path
    files: tuple[Path, ...]
    unique_files: tuple[Path, ...]
    duplicates: dict[Path, tuple[Path, ...]]

    @property
    def duplicate_count(self) -> int:
        return len(self.files) - len(self.unique_files)

    def summary(self) -> str:
        return (
            f"{self.directory}: {len(self.unique_files)} unique image(s), "
            f"{self.duplicate_count} exact duplicate(s)"
        )


def image_files(directory: Path | str) -> tuple[Path, ...]:
    path = Path(directory)
    if not path.exists():
        return ()
    return tuple(
        sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_reference_directory(directory: Path | str) -> DatasetAudit:
    directory = Path(directory)
    files = image_files(directory)
    groups: dict[str, list[Path]] = {}
    for path in files:
        groups.setdefault(sha256_file(path), []).append(path)

    unique_files = tuple(group[0] for group in groups.values())
    duplicates = {
        group[0]: tuple(group[1:])
        for group in groups.values()
        if len(group) > 1
    }
    return DatasetAudit(
        directory=directory,
        files=files,
        unique_files=unique_files,
        duplicates=duplicates,
    )


def validate_reference_sets(
    subjects: Iterable[object],
    minimum_unique_images: int = 1,
) -> tuple[DatasetAudit, ...]:
    audits = tuple(
        audit_reference_directory(subject.reference_dir)
        for subject in subjects
    )
    problems = [
        audit.summary()
        for audit in audits
        if len(audit.unique_files) < minimum_unique_images
    ]
    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(
            "Reference-image validation failed. Add more unique images:\n"
            f"{details}"
        )
    return audits
