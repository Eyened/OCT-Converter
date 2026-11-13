"""DICOM Ophthalmic OCT B-scan Volume Analysis (flow / OCTA) writer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.uid import (
    OphthalmicOpticalCoherenceTomographyBscanVolumeAnalysisStorage,
    OphthalmicTomographyImageStorage,
    generate_uid,
)

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


def _meta() -> DicomMetadata:
    patient = PatientMeta()
    patient.first_name = "Test"
    patient.last_name = "Patient"
    patient.patient_id = "P1"
    series = SeriesMeta()
    series.laterality = "L"
    series.acquisition_date = datetime(2023, 2, 23, 14, 26, 53)
    series.octa = True
    series.description = "OCTA"
    series.scan_pattern = ("DCM", "128279", "Cube B-scan pattern")
    manu = ManufacturerMeta()
    manu.manufacturer = "Topcon"
    manu.manufacturer_model = "DRI OCT Triton plus"
    geom = ImageGeometry()
    geom.pixel_spacing = [0.002621, 0.009375]
    geom.slice_thickness = 0.009404
    geom.image_orientation = [1, 0, 0, 0, 1, 0]
    return DicomMetadata(
        patient_info=patient,
        series_info=series,
        manufacturer_info=manu,
        image_geometry=geom,
        oct_image_params=OCTImageParams(),
    )


def _write_opt(tmp_path: Path, n_frames: int = 4) -> pydicom.Dataset:
    frames = [np.full((8, 6), i, dtype=np.uint8) for i in range(n_frames)]
    path = tmp_path / "struct.dcm"
    return write_opt_dicom(
        _meta(),
        frames,
        path,
        study_instance_uid=generate_uid(),
        frame_of_reference_uid=generate_uid(),
    )


def test_volume_analysis_sop_class_and_modality(tmp_path: Path):
    opt = _write_opt(tmp_path)
    flow = [np.full((8, 6), 3, dtype=np.uint8) for _ in range(4)]
    out = tmp_path / "flow.dcm"
    ds = write_oct_bscan_volume_analysis_dicom(
        _meta(),
        flow,
        opt,
        out,
        bscans_per_frame=4,
    )
    assert Path(out).is_file()
    # Supplement 197 Table A.1-2 / B.5
    assert str(ds.SOPClassUID) == "1.2.840.10008.5.1.4.1.1.77.1.5.8"
    assert ds.SOPClassUID == OphthalmicOpticalCoherenceTomographyBscanVolumeAnalysisStorage
    assert ds.file_meta.MediaStorageSOPClassUID == ds.SOPClassUID
    assert str(opt.SOPClassUID) == "1.2.840.10008.5.1.4.1.1.77.1.5.4"
    assert opt.SOPClassUID == OphthalmicTomographyImageStorage
    assert opt.OphthalmicVolumetricPropertiesFlag == "YES"
    assert ds.Modality == "OPTBSV"
    assert ds.NumberOfFrames == 4
    assert ds.Rows == 8
    assert ds.Columns == 6
    assert ds.SamplesPerPixel == 1
    assert ds.PhotometricInterpretation == "MONOCHROME2"
    assert ds.PixelRepresentation == 1
    assert "WindowCenter" not in ds  # VOI LUT Module forbidden on this IOD
    assert "FrameVOILUTSequence" in ds.SharedFunctionalGroupsSequence[0]
    assert ds.FrameOfReferenceUID == opt.FrameOfReferenceUID
    assert ds.StudyInstanceUID == opt.StudyInstanceUID
    assert ds.SeriesInstanceUID != opt.SeriesInstanceUID


def test_volume_analysis_derivation_links_opt_frames(tmp_path: Path):
    opt = _write_opt(tmp_path, n_frames=3)
    flow = [np.zeros((8, 6), dtype=np.uint8) for _ in range(3)]
    ds = write_oct_bscan_volume_analysis_dicom(
        _meta(), flow, opt, tmp_path / "flow.dcm", bscans_per_frame=4
    )
    assert len(ds.PerFrameFunctionalGroupsSequence) == 3
    for i, fg in enumerate(ds.PerFrameFunctionalGroupsSequence):
        deriv = fg.DerivationImageSequence[0]
        assert deriv.DerivationCodeSequence[0].CodeValue == "128303"
        src = deriv.SourceImageSequence[0]
        assert src.ReferencedSOPClassUID == OphthalmicTomographyImageStorage
        assert src.ReferencedSOPInstanceUID == opt.SOPInstanceUID
        assert int(src.ReferencedFrameNumber) == i + 1
        assert src.PurposeOfReferenceCodeSequence[0].CodeValue == "128250"
        assert src.SpatialLocationsPreserved == "YES"

        ref = fg.ReferencedImageSequence[0]
        assert ref.ReferencedSOPInstanceUID == opt.SOPInstanceUID
        assert int(ref.ReferencedFrameNumber) == i + 1


def test_volume_analysis_acquisition_params(tmp_path: Path):
    opt = _write_opt(tmp_path)
    flow = [np.zeros((8, 6), dtype=np.uint8) for _ in range(4)]
    ds = write_oct_bscan_volume_analysis_dicom(
        _meta(), flow, opt, tmp_path / "flow.dcm", bscans_per_frame=4
    )
    params = ds.OCTBscanAnalysisAcquisitionParametersSequence[0]
    assert params.NumberOfBscansPerFrame == 4
    assert float(params.BscanSlabThickness) == pytest.approx(0.009404, rel=1e-4)
    assert float(params.DistanceBetweenBscanSlabs) == pytest.approx(0.009404, rel=1e-4)
    assert params.ScanPatternTypeCodeSequence[0].CodeValue == "128279"
    family = ds.AcquisitionMethodAlgorithmSequence[0].AlgorithmFamilyCodeSequence[0]
    assert family.CodeValue == "128304"


def test_create_dicom_from_fda_writes_octa_companion(tmp_path: Path):
    import os

    from oct_converter.dicom.dicom import create_dicom_from_fda

    raw = os.environ.get("OCT_CONVERTER_TEST_FDA_OCTA")
    if not raw:
        pytest.skip(
            "Set OCT_CONVERTER_TEST_FDA_OCTA to a .fda path to run this "
            "integration test"
        )
    path = Path(raw)
    if not path.is_file():
        pytest.skip(f"OCT_CONVERTER_TEST_FDA_OCTA does not point to a file: {path}")

    files = create_dicom_from_fda(str(path), str(tmp_path))
    stem = path.stem
    stems = {Path(p).name for p in files}
    assert f"{stem}.dcm" in stems
    assert f"{stem}_octa.dcm" in stems
    opt = pydicom.dcmread(tmp_path / f"{stem}.dcm", stop_before_pixels=True)
    octa = pydicom.dcmread(tmp_path / f"{stem}_octa.dcm", stop_before_pixels=True)
    assert str(opt.SOPClassUID) == "1.2.840.10008.5.1.4.1.1.77.1.5.4"
    assert str(octa.SOPClassUID) == "1.2.840.10008.5.1.4.1.1.77.1.5.8"
    assert octa.SOPClassUID == OphthalmicOpticalCoherenceTomographyBscanVolumeAnalysisStorage
    assert octa.Modality == "OPTBSV"
    assert opt.OphthalmicVolumetricPropertiesFlag == "YES"
    assert octa.NumberOfFrames == opt.NumberOfFrames
    assert octa.NumberOfFrames > 0
    assert octa.FrameOfReferenceUID == opt.FrameOfReferenceUID
    assert octa.StudyInstanceUID == opt.StudyInstanceUID
