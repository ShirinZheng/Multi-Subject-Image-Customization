from __future__ import annotations

from pathlib import Path

from src.config import SubjectConfig
from src.trainer import (
    build_training_command,
    curated_reference_directory,
    resolve_training_script,
    run_training,
)


def test_training_command_uses_argument_list_without_shell_quotes(tmp_path: Path) -> None:
    subject = SubjectConfig(
        name="test",
        class_name="red ceramic cup",
        description="a red cup",
        reference_dir=tmp_path / "references",
        region=(0.1, 0.1, 0.4, 0.8),
        trigger_token="abc",
        lora_dir=tmp_path / "lora",
    )
    script = tmp_path / "train.py"
    data_dir = tmp_path / "curated data"

    command = build_training_command(
        subject,
        script=script,
        instance_data_dir=data_dir,
    )

    assert command[:3] == ["accelerate", "launch", str(script)]
    assert f"--instance_data_dir={data_dir}" in command
    assert "--instance_prompt=a photo of abc red ceramic cup" in command
    assert all("'" not in argument and '"' not in argument for argument in command)


def test_resolve_and_curate_training_inputs(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "train.py"
    script.write_text("# test", encoding="utf-8")
    monkeypatch.setenv("DIFFUSERS_TRAIN_SCRIPT", str(script))
    assert resolve_training_script() == script.resolve()

    references = tmp_path / "references"
    references.mkdir()
    (references / "01.jpg").write_bytes(b"first")
    (references / "02.jpg").write_bytes(b"second")
    (references / "03.jpg").write_bytes(b"first")
    subject = SubjectConfig(
        name="test",
        class_name="test object",
        description="test",
        reference_dir=references,
        region=(0.1, 0.1, 0.4, 0.8),
        lora_dir=tmp_path / "lora",
    )

    with curated_reference_directory(subject) as curated:
        assert len(list(curated.iterdir())) == 2


def test_run_training_executes_argument_list(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = tmp_path / "train.py"
    script.write_text("# test", encoding="utf-8")
    monkeypatch.setenv("DIFFUSERS_TRAIN_SCRIPT", str(script))
    references = tmp_path / "references"
    references.mkdir()
    (references / "01.jpg").write_bytes(b"first")
    (references / "02.jpg").write_bytes(b"second")
    subject = SubjectConfig(
        name="test",
        class_name="test object",
        description="test",
        reference_dir=references,
        region=(0.1, 0.1, 0.4, 0.8),
        lora_dir=tmp_path / "lora",
    )
    calls = []

    def fake_run(command, *, check):
        calls.append((command, check))

    monkeypatch.setattr("src.trainer.subprocess.run", fake_run)
    run_training([subject], force=True)

    assert len(calls) == 1
    assert calls[0][0][0:2] == ["accelerate", "launch"]
    assert calls[0][1] is True
