from __future__ import annotations

from pathlib import Path

from src.data_quality import audit_reference_directory


def test_audit_groups_exact_duplicates(tmp_path: Path) -> None:
    (tmp_path / "01.jpg").write_bytes(b"first")
    (tmp_path / "02.jpg").write_bytes(b"second")
    (tmp_path / "03.jpg").write_bytes(b"first")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    audit = audit_reference_directory(tmp_path)

    assert len(audit.files) == 3
    assert len(audit.unique_files) == 2
    assert audit.duplicate_count == 1
    assert audit.duplicates[tmp_path / "01.jpg"] == (tmp_path / "03.jpg",)
