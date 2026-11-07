"""Tests for the MLflow helper functions that do not need a tracking server."""
from visionguard.tracking import dvc_data_version, git_info, run_tags


def test_dvc_data_version_reads_the_md5_from_the_pointer_file(tmp_path):
    pointer = tmp_path / "raw.dvc"
    pointer.write_text("outs:\n- md5: abc123.dir\n  size: 10\n  nfiles: 2\n  path: raw\n")
    assert dvc_data_version(pointer) == "abc123.dir"


def test_dvc_data_version_is_unknown_when_the_file_is_missing(tmp_path):
    assert dvc_data_version(tmp_path / "missing.dvc") == "unknown"


def test_git_info_has_both_fields():
    info = git_info()
    assert set(info) == {"git_commit", "git_dirty"}
    assert info["git_dirty"] in {"true", "false"}


def test_run_tags_include_the_seed_as_text():
    tags = run_tags(seed=42)
    assert tags["seed"] == "42"
    assert {"git_commit", "git_dirty", "dvc_data_version", "seed"} <= set(tags)
