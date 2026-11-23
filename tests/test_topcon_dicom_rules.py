"""FDA/FDS DICOM rules that follow the standard rather than a vendor file."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from oct_converter.dicom.metadata import (
    DicomMetadata,
    ImageGeometry,
    ManufacturerMeta,
    OCTImageParams,
    PatientMeta,
    SeriesMeta,
)
from oct_converter.readers.scan_geometry import (
    SCAN_PATTERN_CIRCLE,
    SCAN_PATTERN_CROSS,
    SCAN_PATTERN_CUBE,
    SCAN_PATTERN_LINE,
    SCAN_PATTERN_RADIAL,
    topcon_pixel_spacing,
    topcon_scan_pattern,
)


def _meta(
    *,
    scan_pattern=None,
    scan_pattern_specified: bool = False,
    slice_thickness: float = 0.0,
    spacing_between_slices: float = 0.0,
    octa: bool = False,
    description: str = "",
    series_id: str = "",
    patient_id: str = "",
    first_name: str = "",
    last_name: str = "",
) -> DicomMetadata:
    geom = ImageGeometry(
        pixel_spacing=[0.002621, 0.02],
        slice_thickness=slice_thickness,
        image_orientation=[1, 0, 0, 0, 1, 0],
        spacing_between_slices=spacing_between_slices,
    )
    return DicomMetadata(
        patient_info=PatientMeta(
            first_name=first_name,
            last_name=last_name,
            patient_id=patient_id,
        ),
        series_info=SeriesMeta(
            laterality="R",
            acquisition_date=datetime(2024, 1, 2, 3, 4, 5),
            description=description,
            series_id=series_id,
            scan_pattern=scan_pattern,
            scan_pattern_specified=scan_pattern_specified,
            octa=octa,
        ),
        manufacturer_info=ManufacturerMeta(manufacturer="Topcon"),
        image_geometry=geom,
        oct_image_params=OCTImageParams(),
    )


def test_vertical_cube_swaps_lateral_and_stack_spacing():
    vertical = topcon_pixel_spacing(6, 512, 256, 12, 9, 2.621)
    horizontal = topcon_pixel_spacing(2, 512, 256, 12, 9, 2.621)

    assert vertical[0] == pytest.approx(9 / 512)
    assert vertical[1] == pytest.approx(12 / 255)
    assert vertical[2] == pytest.approx(0.002621)
    assert horizontal[0] == pytest.approx(12 / 512)
    assert horizontal[1] == pytest.approx(9 / 255)
    assert horizontal[2] == pytest.approx(0.002621)


def test_cross_gap_is_spacing_between_slices_not_thickness():
    from oct_converter.dicom.dicom import opt_shared_functional_groups

    gap = topcon_pixel_spacing(11, 512, 5, 6, 1.0, 2.6)
    assert gap[1] == pytest.approx(0.25)
    from oct_converter.dicom.dicom import _apply_pixel_measures
    from oct_converter.dicom.fda_meta import topcon_image_geometry

    geom = topcon_image_geometry(gap, 11)
    assert geom.slice_thickness == 0
    assert geom.spacing_between_slices == pytest.approx(0.25)
    horizontal = topcon_image_geometry(
        topcon_pixel_spacing(2, 512, 256, 12, 9, 2.621), 2
    )
    assert horizontal.slice_thickness == pytest.approx(9 / 255)
    assert horizontal.spacing_between_slices == 0

    cross = _meta(
        spacing_between_slices=geom.spacing_between_slices,
        slice_thickness=geom.slice_thickness,
    )
    measures = __import__("pydicom").Dataset()
    _apply_pixel_measures(measures, cross)
    assert "SliceThickness" not in measures
    assert float(measures.SpacingBetweenSlices) == pytest.approx(0.25)

    line = _meta(slice_thickness=0.0, spacing_between_slices=0.0)
    line_measures = __import__("pydicom").Dataset()
    _apply_pixel_measures(line_measures, line)
    assert "SliceThickness" not in line_measures
    assert "SpacingBetweenSlices" not in line_measures

    zero = topcon_pixel_spacing(11, 512, 5, 6, 0, 2.6)
    assert zero[1] == 0.0
    radial = topcon_pixel_spacing(3, 1024, 12, 6, 6, 2.6)
    assert radial[1] == 0.0
    single = topcon_pixel_spacing(4, 512, 1, 6, 1.0, 2.6)
    assert single[1] == 0.0


@pytest.mark.parametrize(
    "mode,expected",
    [
        (0, SCAN_PATTERN_LINE),
        (1, SCAN_PATTERN_CIRCLE),
        (2, SCAN_PATTERN_CUBE),
        (3, SCAN_PATTERN_RADIAL),
        (4, SCAN_PATTERN_CROSS),
        (6, SCAN_PATTERN_CUBE),
        (11, SCAN_PATTERN_CROSS),
    ],
)
def test_known_scan_modes_keep_their_codes(mode, expected):
    assert topcon_scan_pattern(mode, num_frames=19) == expected


def test_unrecognised_multiframe_mode_is_not_cube():
    assert topcon_scan_pattern(19, num_frames=19) != SCAN_PATTERN_CUBE
    assert topcon_scan_pattern(7, num_frames=7) is None


def test_structural_octa_image_type_and_series_description(tmp_path: Path):
    from oct_converter.dicom.fda_meta import opt_series_fallback
    from oct_converter.dicom.study import DicomStudy, ExportOct
    from oct_converter.dicom.write import write_dicom_study

    assert opt_series_fallback(2, octa=True) == "OCT"
    assert opt_series_fallback(6, octa=True) == "OCT"
    meta = _meta(
        scan_pattern=SCAN_PATTERN_CUBE,
        slice_thickness=0.047059,
        octa=True,
        description="",
    )
    flow = [np.zeros((4, 4), dtype=np.uint8) for _ in range(2)]
    study = DicomStudy(
        series=[
            ExportOct(
                meta=meta,
                filename="struct.dcm",
                volume=[np.zeros((4, 4), dtype=np.uint8) for _ in range(2)],
                description_fallback=opt_series_fallback(2, True),
                angiography=flow,
                angiography_filename="flow.dcm",
                angiography_description="OCTA flow",
            )
        ]
    )
    write_dicom_study(study, tmp_path)
    import pydicom

    opt = pydicom.dcmread(tmp_path / "struct.dcm", stop_before_pixels=True)
    octa = pydicom.dcmread(tmp_path / "flow.dcm", stop_before_pixels=True)
    assert list(opt.ImageType) == ["ORIGINAL", "PRIMARY"]
    assert opt.SeriesDescription == "OCT"
    assert opt.OphthalmicVolumetricPropertiesFlag == "YES"
    assert str(octa.SOPClassUID) == "1.2.840.10008.5.1.4.1.1.77.1.5.8"
    assert octa.SeriesDescription == "OCTA flow"


def test_volumetric_flag_is_no_for_a_line(tmp_path: Path):
    from oct_converter.dicom.dicom import write_opt_dicom

    path = tmp_path / "line.dcm"
    ds = write_opt_dicom(
        _meta(scan_pattern=SCAN_PATTERN_LINE, slice_thickness=0.0),
        [np.zeros((4, 4), dtype=np.uint8)],
        path,
    )
    assert ds.OphthalmicVolumetricPropertiesFlag == "NO"


def test_caller_patient_fills_only_empty_fda_fields(tmp_path: Path):
    from oct_converter.dicom.study import DicomStudy, ExportOct
    from oct_converter.dicom.write import write_dicom_study

    empty = _meta(patient_id="", last_name="", scan_pattern=SCAN_PATTERN_LINE)
    present = _meta(
        patient_id="FDA-1",
        last_name="FromFile",
        first_name="Ada",
        scan_pattern=SCAN_PATTERN_LINE,
    )
    supplied = PatientMeta(
        patient_id="CALLER",
        last_name="Other",
        first_name="Bea",
        patient_sex="F",
        patient_dob=datetime(1970, 4, 5),
    )
    frames = [np.zeros((4, 4), dtype=np.uint8)]
    write_dicom_study(
        DicomStudy(
            series=[ExportOct(meta=empty, filename="empty.dcm", volume=frames)]
        ),
        tmp_path,
        patient=supplied,
    )
    write_dicom_study(
        DicomStudy(
            series=[ExportOct(meta=present, filename="present.dcm", volume=frames)]
        ),
        tmp_path,
        patient=supplied,
    )
    import pydicom

    filled = pydicom.dcmread(tmp_path / "empty.dcm", stop_before_pixels=True)
    kept = pydicom.dcmread(tmp_path / "present.dcm", stop_before_pixels=True)
    assert filled.PatientID == "CALLER"
    assert str(filled.PatientName) == "Other^Bea"
    assert filled.PatientSex == "F"
    assert filled.PatientBirthDate == "19700405"
    assert kept.PatientID == "FDA-1"
    assert str(kept.PatientName) == "FromFile^Ada"
    assert empty.patient_info.patient_id == ""


@pytest.mark.parametrize("series_id", ["", "0", 0, None])
def test_series_number_falls_back_to_one(series_id):
    from oct_converter.dicom.dicom import populate_opt_series

    meta = _meta(series_id="" if series_id is None else series_id)
    if series_id is None:
        meta.series_info.series_id = None  # type: ignore[assignment]
    ds = populate_opt_series(__import__("pydicom").Dataset(), meta)
    assert int(ds.SeriesNumber) == 1
    assert ds.StudyID == ""


def test_positive_session_id_is_series_number():
    from oct_converter.dicom.dicom import populate_opt_series

    ds = populate_opt_series(
        __import__("pydicom").Dataset(), _meta(series_id="12")
    )
    assert int(ds.SeriesNumber) == 12


def test_specific_character_set_is_utf8():
    from oct_converter.dicom.dicom import populate_patient_info

    ds = populate_patient_info(
        __import__("pydicom").Dataset(), _meta(last_name="File")
    )
    assert ds.SpecificCharacterSet == "ISO_IR 192"


def test_unrecognised_mode_omits_scan_pattern_code(tmp_path: Path):
    from oct_converter.dicom.dicom import write_opt_dicom

    ds = write_opt_dicom(
        _meta(scan_pattern=None, scan_pattern_specified=True),
        [np.zeros((4, 4), dtype=np.uint8) for _ in range(3)],
        tmp_path / "web.dcm",
    )
    assert "ScanPatternTypeCodeSequence" not in ds
    assert ds.OphthalmicVolumetricPropertiesFlag == "NO"
