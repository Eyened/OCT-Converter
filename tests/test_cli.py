"""CLI for proprietary OCT → DICOM conversion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from oct_converter.cli import collect_oct_inputs, main, output_dir_for


SUPPORTED = {".fda", ".fds", ".e2e", ".img", ".oct"}


def test_collect_oct_inputs_single_file(tmp_path: Path):
    fda = tmp_path / "a.fda"
    fda.write_bytes(b"x")
    assert collect_oct_inputs([fda]) == [fda.resolve()]


def test_collect_oct_inputs_directory(tmp_path: Path):
    (tmp_path / "one.fda").write_bytes(b"1")
    (tmp_path / "two.FDS").write_bytes(b"2")
    (tmp_path / "skip.txt").write_bytes(b"3")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "three.e2e").write_bytes(b"4")

    found = collect_oct_inputs([tmp_path], recursive=True)
    names = {p.name.lower() for p in found}
    assert names == {"one.fda", "two.fds", "three.e2e"}


def test_collect_oct_inputs_rejects_unknown(tmp_path: Path):
    bad = tmp_path / "note.txt"
    bad.write_bytes(b"x")
    with pytest.raises(SystemExit):
        collect_oct_inputs([bad])


def test_output_dir_for_single_uses_target(tmp_path: Path):
    assert output_dir_for(tmp_path / "a.fda", tmp_path / "out", batch=False) == (
        tmp_path / "out"
    )


def test_output_dir_for_batch_uses_stem_subdir(tmp_path: Path):
    assert output_dir_for(tmp_path / "a.fda", tmp_path / "out", batch=True) == (
        tmp_path / "out" / "a"
    )


def test_main_converts_file(tmp_path: Path):
    fda = tmp_path / "sample.fda"
    fda.write_bytes(b"FOCT")
    out = tmp_path / "dicom"
    written = [out / "sample.dcm", out / "sample_octa.dcm"]

    with patch("oct_converter.cli.create_dicom_from_oct", return_value=written) as conv:
        code = main(["-o", str(out), str(fda)])
    assert code == 0
    conv.assert_called_once_with(str(fda.resolve()), output_dir=str(out))


def test_main_batch_directory(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.fda").write_bytes(b"1")
    (src / "b.fda").write_bytes(b"2")
    out = tmp_path / "dicom"

    with patch(
        "oct_converter.cli.create_dicom_from_oct",
        side_effect=lambda path, output_dir: [Path(output_dir) / "x.dcm"],
    ) as conv:
        code = main(["-o", str(out), str(src)])
    assert code == 0
    assert conv.call_count == 2
    called_dirs = sorted(c.kwargs["output_dir"] for c in conv.call_args_list)
    assert called_dirs == [str(out / "a"), str(out / "b")]
