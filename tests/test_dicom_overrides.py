"""Write-time DICOM overrides stay off the parsed metadata."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

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


def _meta(patient_id: str = "file-id") -> DicomMetadata:
    return DicomMetadata(
        patient_info=PatientMeta(first_name="Ann", last_name="File", patient_id=patient_id),
        series_info=SeriesMeta(
            laterality="L",
            acquisition_date=datetime(2020, 5, 4, 3, 2, 1),
            description="from file",
            protocol="cube",
        ),
        manufacturer_info=ManufacturerMeta(manufacturer="Topcon"),
        image_geometry=ImageGeometry(
            pixel_spacing=[0.01, 0.02],
            slice_thickness=0.03,
            image_orientation=[1, 0, 0, 0, 1, 0],
        ),
        oct_image_params=OCTImageParams(),
    )


def test_prepare_export_copies_metadata_and_applies_tags():
    from oct_converter.dicom.export import prepare_export

    source = _meta()
    prepared = prepare_export(
        source,
        {
            "PatientName": "Doe^Jane",
            "PatientID": "P001",
            "PatientSex": "F",
            "PatientBirthDate": "19800102",
            "Laterality": "R",
            "SeriesDescription": "custom",
            "Manufacturer": "Acme",
            "StudyInstanceUID": "1.2.3.4.5",
            "SeriesInstanceUID": {"opt": "1.2.3.4.6", "fundus": ["1.2.3.4.7"]},
        },
    )

    assert source.patient_info.patient_id == "file-id"
    assert source.patient_info.last_name == "File"
    assert source.series_info.laterality == "L"
    assert prepared.meta.patient_info.last_name == "Doe"
    assert prepared.meta.patient_info.first_name == "Jane"
    assert prepared.meta.patient_info.patient_id == "P001"
    assert prepared.meta.patient_info.patient_sex == "F"
    assert prepared.meta.patient_info.patient_dob == datetime(1980, 1, 2)
    assert prepared.meta.series_info.laterality == "R"
    assert prepared.meta.series_info.description == "custom"
    assert prepared.meta.manufacturer_info.manufacturer == "Acme"
    assert prepared.study_instance_uid == "1.2.3.4.5"
    assert prepared.series.for_role("opt") == "1.2.3.4.6"
    assert prepared.series.for_role("fundus", 0) == "1.2.3.4.7"
    assert prepared.series.for_role("octa") is None


def test_string_series_uid_is_only_the_first_of_that_role():
    from oct_converter.dicom.export import DicomOverrideError, prepare_export

    prepared = prepare_export(_meta(), {"SeriesInstanceUID": "1.2.3"})
    assert prepared.series.for_role("opt", 0) == "1.2.3"
    with pytest.raises(DicomOverrideError):
        prepared.series.for_role("opt", 1)


def test_unknown_override_is_rejected():
    from oct_converter.dicom.export import DicomOverrideError, prepare_export

    with pytest.raises(DicomOverrideError, match="NotATag"):
        prepare_export(_meta(), {"NotATag": "x"})


def test_fda_dicom_metadata_returns_independent_instances():
    from oct_converter.dicom.fda_meta import fda_dicom_metadata

    def fake(patient_id: str):
        return SimpleNamespace(
            metadata={
                "patient_info_02": {
                    "patient_id": patient_id,
                    "first_name": "A",
                    "last_name": "B",
                    "sex": 1,
                    "birth_date": (1980, 1, 2),
                },
                "capture_info_02": {
                    "session_id": "7",
                    "eye": 0,
                    "cap_date": (2020, 1, 2, 3, 4, 5),
                    "label": "macula",
                },
            },
            header={},
            pixel_spacing=[0.01, 0.003, 0.03],
            oct_header={"number_slices": 1, "scan_mode": 2},
            volume=[np.zeros((4, 4), dtype=np.uint8)],
        )

    first = fda_dicom_metadata(fake("one"))
    second = fda_dicom_metadata(fake("two"))
    assert first.patient_info.patient_id == "one"
    assert second.patient_info.patient_id == "two"


def test_create_dicom_from_fda_writes_overrides_without_editing_meta(tmp_path, monkeypatch):
    from oct_converter.dicom import dicom as dicom_mod

    volume = [np.full((8, 8), 3, dtype=np.uint8)]
    parsed = SimpleNamespace(
        metadata={
            "patient_info_02": {
                "patient_id": "from-fda",
                "first_name": "Ann",
                "last_name": "File",
                "sex": 2,
                "birth_date": (1970, 6, 7),
            },
            "capture_info_02": {
                "session_id": "9",
                "eye": 1,
                "cap_date": (2021, 2, 3, 4, 5, 6),
                "label": "optic",
            },
        },
        header={},
        pixel_spacing=[0.01, 0.003, 0.03],
        oct_header={"number_slices": 1, "scan_mode": 2, "height": 8},
        volume=volume,
        contours=None,
    )

    class FakeFDA:
        def __init__(self, path):
            self.path = path

        def read_oct_volume(self, verbose=False, *, decode=True):
            return parsed

        def read_fundus_image_bytes(self):
            return None

        def read_angiography_volume(self):
            return None

        def read_fundus_image_gray_scale_bytes(self):
            return None

        def read_fundus_image_gray_scale(self):
            return None

    from oct_converter.dicom.fda_meta import fda_dicom_metadata

    monkeypatch.setattr("oct_converter.dicom.parse.FDA", FakeFDA)
    supplied = fda_dicom_metadata(parsed)
    paths = dicom_mod.create_dicom_from_fda(
        "scan.fda",
        str(tmp_path),
        meta=supplied,
        overrides={
            "PatientID": "P001",
            "PatientName": "Doe^Jane",
            "StudyInstanceUID": "1.2.826.0.1.3680043.10.1",
            "SeriesInstanceUID": {"opt": "1.2.826.0.1.3680043.10.2"},
        },
    )

    assert supplied.patient_info.patient_id == "from-fda"
    assert supplied.patient_info.last_name == "File"
    import pydicom

    ds = pydicom.dcmread(paths[0])
    assert ds.PatientID == "P001"
    assert str(ds.PatientName) == "Doe^Jane"
    assert ds.StudyInstanceUID == "1.2.826.0.1.3680043.10.1"
    assert ds.SeriesInstanceUID == "1.2.826.0.1.3680043.10.2"


def test_write_dicom_study_does_not_depend_on_source_format(tmp_path):
    from oct_converter.dicom.study import DicomStudy, ExportOct
    from oct_converter.dicom.write import write_dicom_study

    frames = [np.full((8, 8), 4, dtype=np.uint8)]
    study = DicomStudy(
        series=[
            ExportOct(
                meta=_meta("parsed-id"),
                filename="volume.dcm",
                volume=frames,
            )
        ]
    )
    paths = write_dicom_study(
        study,
        tmp_path,
        overrides={"PatientID": "P001", "StudyInstanceUID": "1.2.826.0.1.3680043.10.9"},
    )
    import pydicom

    ds = pydicom.dcmread(paths[0])
    assert study.series[0].meta.patient_info.patient_id == "parsed-id"
    assert ds.PatientID == "P001"
    assert ds.StudyInstanceUID == "1.2.826.0.1.3680043.10.9"
    assert ds.Modality == "OPT"

