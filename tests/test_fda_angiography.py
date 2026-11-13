"""Parse Topcon FDA @IMG_ANGIOGRAPHY_02 into a B-scan volume."""

from __future__ import annotations

import io
import os
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from oct_converter.readers.fda import FDA, parse_angiography_chunk


def _optional_fda(env_var: str) -> Path:
    """Return a path from ``env_var``, or skip when unset / missing."""
    raw = os.environ.get(env_var)
    if not raw:
        pytest.skip(f"Set {env_var} to a .fda path to run this integration test")
    path = Path(raw)
    if not path.is_file():
        pytest.skip(f"{env_var} does not point to a file: {path}")
    return path


def _angio_header(width: int, height: int, n_slices: int) -> bytes:
    # Same layout as fda_binary.oct_header_2
    return struct.pack(
        "<BIIIIBI",
        2,  # scan_mode
        width,
        height,
        8,  # bits_per_pixel
        n_slices,
        4,  # format (zip chunks)
        width * height * n_slices,
    )


def _zip_angio(frames: np.ndarray) -> bytes:
    """One Topcon IMAGEnet-style zip containing a raw uint8 ``angio`` blob."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("angio", np.ascontiguousarray(frames, dtype=np.uint8).tobytes())
    return buf.getvalue()


def test_parse_angiography_chunk_concatenates_zip_slices():
    height, width = 4, 3
    chunk0 = np.arange(2 * height * width, dtype=np.uint8).reshape(2, height, width)
    chunk1 = (chunk0 + 50).astype(np.uint8)
    # 8-byte preamble before first PK (ignored except for locating zips)
    preamble = b"\x00\x1e" + b"\x00" * 6
    payload = preamble + _zip_angio(chunk0) + _zip_angio(chunk1)
    raw = _angio_header(width, height, 4) + payload

    frames = parse_angiography_chunk(raw)

    assert len(frames) == 4
    assert frames[0].shape == (height, width)
    assert frames[0].dtype == np.uint8
    np.testing.assert_array_equal(frames[0], chunk0[0])
    np.testing.assert_array_equal(frames[2], chunk1[0])
    np.testing.assert_array_equal(frames[3], chunk1[1])


def test_parse_angiography_chunk_rejects_empty():
    with pytest.raises(ValueError, match="header"):
        parse_angiography_chunk(b"\x00" * 10)


def test_read_angiography_volume_returns_none_without_chunk():
    path = _optional_fda("OCT_CONVERTER_TEST_FDA")
    fda = FDA(path)
    assert b"@IMG_ANGIOGRAPHY_02" not in fda.chunk_dict
    assert fda.read_angiography_volume() is None


def test_read_angiography_volume_on_octa_fda():
    path = _optional_fda("OCT_CONVERTER_TEST_FDA_OCTA")
    fda = FDA(path)
    frames = fda.read_angiography_volume()
    assert frames is not None
    assert len(frames) > 0
    assert frames[0].ndim == 2
    assert frames[0].dtype == np.uint8
    stacked = np.stack(frames, axis=0)
    assert stacked.max() > 0
