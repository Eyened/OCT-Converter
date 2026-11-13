"""Lossless JPEG 2000 encoding for OCT / OCTA pixel frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.uid import JPEG2000Lossless

from oct_converter.dicom.j2k import encode_j2k_lossless_frames, prepare_j2k_frames


def test_encode_j2k_lossless_uint8_roundtrip():
    frames = [np.arange(16, dtype=np.uint8).reshape(4, 4) + i for i in range(3)]
    payloads = encode_j2k_lossless_frames(frames)
    assert len(payloads) == 3
    j2k = prepare_j2k_frames(payloads)
    assert j2k.reversible is True
    assert j2k.rows == 4 and j2k.columns == 4
    assert j2k.bits_allocated == 8
    assert j2k.pixel_representation == 0
    import imagecodecs

    for src, codestream in zip(frames, payloads):
        assert np.array_equal(imagecodecs.jpeg2k_decode(codestream), src)


def test_encode_j2k_lossless_int16_signed():
    frames = [np.array([[0, 1], [2, 77]], dtype=np.int16)]
    payloads = encode_j2k_lossless_frames(frames)
    j2k = prepare_j2k_frames(payloads)
    assert j2k.bits_allocated == 16
    assert j2k.pixel_representation == 1
    assert j2k.reversible is True


def test_write_opt_dicom_compresses_raw_frames(tmp_path: Path):
    from datetime import datetime

    from oct_converter.dicom.dicom import write_opt_dicom
    from oct_converter.dicom.metadata import (
        DicomMetadata,
        ImageGeometry,
        ManufacturerMeta,
        OCTImageParams,
        PatientMeta,
        SeriesMeta,
    )

    meta = DicomMetadata(
        patient_info=PatientMeta(patient_id="1"),
        series_info=SeriesMeta(
            laterality="L",
            acquisition_date=datetime(2023, 1, 1),
            description="OCT",
        ),
        manufacturer_info=ManufacturerMeta(manufacturer="Topcon"),
        image_geometry=ImageGeometry(
            pixel_spacing=[0.01, 0.01],
            slice_thickness=0.01,
            image_orientation=[1, 0, 0, 0, 1, 0],
        ),
        oct_image_params=OCTImageParams(),
    )
    frames = [np.full((32, 24), i, dtype=np.uint8) for i in range(4)]
    path = tmp_path / "opt.dcm"
    ds = write_opt_dicom(meta, frames, path)
    assert ds.file_meta.TransferSyntaxUID == JPEG2000Lossless
    assert ds.LossyImageCompression == "00"
    assert ds["PixelData"].is_undefined_length


def test_volume_analysis_uses_j2k_lossless(tmp_path: Path):
    from datetime import datetime

    from pydicom.uid import generate_uid

    from oct_converter.dicom.dicom import (
        write_oct_bscan_volume_analysis_dicom,
        write_opt_dicom,
    )
    from oct_converter.dicom.metadata import (
        DicomMetadata,
        ImageGeometry,
        ManufacturerMeta,
        OCTImageParams,
        PatientMeta,
        SeriesMeta,
    )

    meta = DicomMetadata(
        patient_info=PatientMeta(patient_id="1"),
        series_info=SeriesMeta(
            laterality="L",
            acquisition_date=datetime(2023, 1, 1),
            octa=True,
            description="OCTA",
            scan_pattern=("DCM", "128279", "Cube B-scan pattern"),
        ),
        manufacturer_info=ManufacturerMeta(manufacturer="Topcon"),
        image_geometry=ImageGeometry(
            pixel_spacing=[0.01, 0.01],
            slice_thickness=0.01,
            image_orientation=[1, 0, 0, 0, 1, 0],
        ),
        oct_image_params=OCTImageParams(),
    )
    frames = [np.zeros((16, 12), dtype=np.uint8) for _ in range(3)]
    opt = write_opt_dicom(
        meta,
        frames,
        tmp_path / "opt.dcm",
        study_instance_uid=generate_uid(),
        frame_of_reference_uid=generate_uid(),
    )
    out = tmp_path / "flow.dcm"
    ds = write_oct_bscan_volume_analysis_dicom(
        meta, frames, opt, out, bscans_per_frame=4
    )
    assert ds.file_meta.TransferSyntaxUID == JPEG2000Lossless
    assert ds.PixelRepresentation == 1
    assert ds.LossyImageCompression == "00"
    assert ds["PixelData"].is_undefined_length
