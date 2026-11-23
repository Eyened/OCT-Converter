"""Colour fundus DICOM uses JPEG 2000 lossless after Topcon BGR correction."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from pydicom.uid import JPEG2000Lossless

from oct_converter.dicom.dicom import write_color_fundus_dicom
from oct_converter.dicom.j2k import encode_j2k_lossless_frames, prepare_j2k_frames
from oct_converter.dicom.metadata import (
    DicomMetadata,
    ImageGeometry,
    ManufacturerMeta,
    OCTImageParams,
    PatientMeta,
    SeriesMeta,
)


def _meta() -> DicomMetadata:
    return DicomMetadata(
        patient_info=PatientMeta(patient_id="1"),
        series_info=SeriesMeta(
            laterality="L",
            acquisition_date=datetime(2023, 1, 1),
            description="COLOR",
        ),
        manufacturer_info=ManufacturerMeta(manufacturer="Topcon"),
        image_geometry=ImageGeometry(
            pixel_spacing=[0.01, 0.01],
            slice_thickness=0.0,
            image_orientation=[1, 0, 0, 0, 1, 0],
        ),
        oct_image_params=OCTImageParams(),
    )


def test_encode_j2k_lossless_rgb_roundtrip():
    import imagecodecs

    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 1] = 20
    rgb[..., 2] = 30
    payloads = encode_j2k_lossless_frames([rgb])
    j2k = prepare_j2k_frames(payloads)
    assert j2k.samples == 3
    assert j2k.reversible is True
    assert j2k.photometric == "RGB"
    decoded = imagecodecs.jpeg2k_decode(payloads[0])
    assert np.array_equal(decoded, rgb)


def test_write_color_fundus_dicom_compresses_rgb(tmp_path: Path):
    rgb = np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8)
    path = tmp_path / "fundus.dcm"
    ds = write_color_fundus_dicom(_meta(), rgb, path)
    assert ds.file_meta.TransferSyntaxUID == JPEG2000Lossless
    assert ds.LossyImageCompression == "00"
    assert ds.SamplesPerPixel == 3
    assert ds.PhotometricInterpretation == "RGB"
    assert ds["PixelData"].is_undefined_length
