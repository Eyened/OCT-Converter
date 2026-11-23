"""Copy ``DicomMetadata`` and apply write-time overrides.

Readers stay format-specific. Each ``*_dicom_metadata`` function turns a
parsed volume into ``DicomMetadata``. Export uses a copy of that object, so
overrides change the DICOM without changing the caller's metadata.
"""

from __future__ import annotations

import copy
import typing as t
from dataclasses import dataclass
from datetime import datetime

from oct_converter.dicom.metadata import DicomMetadata

SERIES_ROLES = ("opt", "fundus", "fundus_grayscale", "octa", "seg")

_SUPPORTED = (
    "PatientName",
    "PatientID",
    "PatientSex",
    "PatientBirthDate",
    "Laterality",
    "StudyID",
    "SeriesNumber",
    "ProtocolName",
    "SeriesDescription",
    "AcquisitionDateTime",
    "Manufacturer",
    "ManufacturerModelName",
    "DeviceSerialNumber",
    "SoftwareVersions",
    "StudyInstanceUID",
    "SeriesInstanceUID",
)


class DicomOverrideError(ValueError):
    """An override keyword or series role is not supported."""


@dataclass(frozen=True)
class SeriesUidPlan:
    """Series Instance UIDs keyed by output role.

    A string is the UID of the first file of that role. A list is one UID
    per file, in the order the exporter writes them. A missing entry means
    the writer generates a UID.
    """

    by_role: dict[str, str | list[str]]

    def for_role(self, role: str, index: int = 0) -> str | None:
        if role not in self.by_role:
            return None
        value = self.by_role[role]
        if isinstance(value, str):
            if index != 0:
                raise DicomOverrideError(
                    f"SeriesInstanceUID role {role!r} is a single UID, "
                    f"but this export writes more than one {role} series. "
                    "Pass a list of UIDs in write order."
                )
            return value
        if index >= len(value):
            return None
        return value[index]


@dataclass(frozen=True)
class PreparedExport:
    meta: DicomMetadata
    study_instance_uid: str | None
    series: SeriesUidPlan


def prepare_export(
    meta: DicomMetadata,
    overrides: t.Mapping[str, t.Any] | None = None,
) -> PreparedExport:
    """Return a copy of ``meta`` with ``overrides`` applied.

    ``StudyInstanceUID`` and ``SeriesInstanceUID`` are returned beside the
    metadata. They are chosen per written file, not stored on ``DicomMetadata``.
    A string ``SeriesInstanceUID`` is the structural OPT series. A dict maps
    ``opt``, ``fundus``, ``fundus_grayscale``, ``octa``, and ``seg`` to a UID
    or a list of UIDs.
    """
    working = copy.deepcopy(meta)
    if not overrides:
        return PreparedExport(working, None, SeriesUidPlan({}))

    pending = dict(overrides)
    unknown = [key for key in pending if key not in _SUPPORTED]
    if unknown:
        supported = ", ".join(_SUPPORTED)
        raise DicomOverrideError(
            f"Unknown override {unknown[0]!r}. Supported keywords: {supported}"
        )

    study_uid = pending.pop("StudyInstanceUID", None)
    series = _series_plan(pending.pop("SeriesInstanceUID", None))
    for keyword, value in pending.items():
        _apply_keyword(working, keyword, value)
    return PreparedExport(working, study_uid, series)


def _series_plan(value: t.Any) -> SeriesUidPlan:
    if value is None:
        return SeriesUidPlan({})
    if isinstance(value, str):
        return SeriesUidPlan({"opt": value})
    if not isinstance(value, dict):
        raise DicomOverrideError(
            "SeriesInstanceUID must be a UID string or a dict of "
            f"{', '.join(SERIES_ROLES)} to a UID or list of UIDs"
        )
    parsed: dict[str, str | list[str]] = {}
    for role, uid in value.items():
        if role not in SERIES_ROLES:
            raise DicomOverrideError(
                f"Unknown SeriesInstanceUID role {role!r}. "
                f"Expected one of {', '.join(SERIES_ROLES)}"
            )
        if isinstance(uid, str):
            parsed[role] = uid
        elif isinstance(uid, (list, tuple)) and all(isinstance(item, str) for item in uid):
            parsed[role] = list(uid)
        else:
            raise DicomOverrideError(
                f"SeriesInstanceUID[{role!r}] must be a UID string or a list of UID strings"
            )
    return SeriesUidPlan(parsed)


def _apply_keyword(meta: DicomMetadata, keyword: str, value: t.Any) -> None:
    patient = meta.patient_info
    series = meta.series_info
    manufacturer = meta.manufacturer_info
    if keyword == "PatientName":
        last, first = _person_name(value)
        patient.last_name = last
        patient.first_name = first
    elif keyword == "PatientID":
        patient.patient_id = "" if value is None else str(value)
    elif keyword == "PatientSex":
        patient.patient_sex = "" if value is None else str(value)
    elif keyword == "PatientBirthDate":
        patient.patient_dob = _parse_date(value, keyword)
    elif keyword == "Laterality":
        series.laterality = "" if value is None else str(value)
    elif keyword == "StudyID":
        series.study_id = "" if value is None else str(value)
    elif keyword == "SeriesNumber":
        series.series_id = "" if value is None else str(value)
    elif keyword == "ProtocolName":
        series.protocol = "" if value is None else str(value)
    elif keyword == "SeriesDescription":
        series.description = "" if value is None else str(value)
    elif keyword == "AcquisitionDateTime":
        series.acquisition_date = _parse_datetime(value)
    elif keyword == "Manufacturer":
        manufacturer.manufacturer = "" if value is None else str(value)
    elif keyword == "ManufacturerModelName":
        manufacturer.manufacturer_model = "" if value is None else str(value)
    elif keyword == "DeviceSerialNumber":
        manufacturer.device_serial = "" if value is None else str(value)
    elif keyword == "SoftwareVersions":
        manufacturer.software_version = "" if value is None else str(value)
    else:
        raise DicomOverrideError(f"Unknown override {keyword!r}")


def _person_name(value: t.Any) -> tuple[str, str]:
    if value is None or value == "":
        return "", ""
    text = str(value)
    family, _, given = text.partition("^")
    return family, given.split("^", 1)[0]


def _parse_date(value: t.Any, keyword: str) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    if len(text) >= 8 and text[:8].isdigit():
        return datetime(int(text[0:4]), int(text[4:6]), int(text[6:8]))
    raise DicomOverrideError(
        f"{keyword} must be a datetime or a DICOM date YYYYMMDD, got {value!r}"
    )


def _parse_datetime(value: t.Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).split(".", 1)[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    try:
        if len(digits) >= 14:
            return datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
        if len(digits) == 8:
            return datetime.strptime(digits, "%Y%m%d")
    except ValueError as exc:
        raise DicomOverrideError(
            f"AcquisitionDateTime must be a datetime or DICOM DT, got {value!r}"
        ) from exc
    raise DicomOverrideError(
        f"AcquisitionDateTime must be a datetime or DICOM DT, got {value!r}"
    )
