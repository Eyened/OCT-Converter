from __future__ import annotations

from datetime import datetime

from oct_converter.dicom.fda_meta import (
    apply_topcon_device_specs,
    topcon_image_geometry,
    _model_name,
)
from oct_converter.dicom.metadata import (
    DicomMetadata,
    ImageGeometry,
    ManufacturerMeta,
    OCTImageParams,
    OPTAcquisitionDevice,
    OPTAnatomyStructure,
    PatientMeta,
    SeriesMeta,
)
from oct_converter.image_types import OCTVolumeWithMetaData
from oct_converter.readers.scan_geometry import topcon_scan_pattern


def fds_patient_meta(fds_metadata: dict) -> PatientMeta:
    """Creates PatientMeta from FDS metadata

    Args:
        fds_metadata: Nested dictionary of metadata collected by the fds reader
    Returns:
        PatientMeta: Patient metadata populated by fds_metadata
    """
    patient_info = fds_metadata.get("patient_info_02") or fds_metadata.get(
        "patient_info", {}
    )
    sex_map = {1: "M", 2: "F", 3: "O", None: ""}
    patient = PatientMeta()

    patient.first_name = patient_info.get("first_name")
    patient.last_name = patient_info.get("last_name")
    patient.patient_id = patient_info.get("patient_id")
    patient.patient_sex = sex_map[patient_info.get("sex", None)]
    try:
        patient.patient_dob = datetime(*patient_info.get("birth_date"))
    except (TypeError, ValueError):
        pass

    return patient


def fds_series_meta(fds_metadata: dict) -> SeriesMeta:
    """Creates SeriesMeta from FDS metadata

    Args:
        fds_metadata: Nested dictionary of metadata collected by the fds reader
    Returns:
        SeriesMeta: Series metadata populated by fds_metadata
    """
    capture_info = fds_metadata.get("capture_info_02") or fds_metadata.get(
        "capture_info", {}
    )
    lat_map = {0: "R", 1: "L", None: ""}
    series = SeriesMeta()

    series.study_id = ""
    series.series_id = capture_info.get("session_id", "")
    series.laterality = lat_map[capture_info.get("eye", None)]
    series.acquisition_date = datetime(*capture_info.get("cap_date"))
    series.opt_anatomy = OPTAnatomyStructure.Retina

    return series


def fds_manu_meta(fds_metadata: dict, fds_header: dict) -> ManufacturerMeta:
    """Creates ManufacturerMeta from FDS metadata

    Args:
        fds_metadata: Nested dictionary of metadata collected by the fds reader
        fds_header: Dictionary of file header information parsed by the fda reader
    Returns:
        ManufacturerMeta: Manufacture metadata populated by fds_metadata
    """
    hw_info = (
        fds_metadata.get("hw_info_03")
        or fds_metadata.get("hw_info_02")
        or fds_metadata.get("hw_info_01", {})
    )
    manufacture = ManufacturerMeta()

    manufacture.manufacturer = "Topcon"
    manufacture.manufacturer_model = _model_name(hw_info.get("model_name"))
    manufacture.device_serial = _model_name(hw_info.get("serial_number"))
    manufacture.software_version = (
        f"{fds_header.get('major_ver')}.{fds_header.get('minor_ver')}"
    )

    return manufacture


def fds_image_geom(pixel_spacing: list, scan_mode: int | None = None) -> ImageGeometry:
    """Creates ImageGeometry from FDS metadata.

    Same axis rules as FDA: vertical cubes swap the field sizes, and a
    cross-scan gap is not stored as slice thickness.
    """
    return topcon_image_geometry(pixel_spacing, scan_mode)


def fds_image_params(model_name: str) -> OCTImageParams:
    """OCT parameters for the model named in the FDS hardware chunk.

    Same conservative Topcon policy as FDA: nominal wavelength may be written
    when the model matches; brochure spatial resolution / half-bandwidth are
    kept on ``manufacturer_specs`` until DICOM equivalence is established.
    """
    image_params = OCTImageParams()
    image_params.opt_acquisition_device = OPTAcquisitionDevice.OCTScanner
    apply_topcon_device_specs(image_params, model_name)
    return image_params


def fds_dicom_metadata(oct: OCTVolumeWithMetaData) -> DicomMetadata:
    """Creates DicomMetadata and populates each module

    Args:
        oct: OCTVolumeWithMetaData created by the fds reader
    Returns:
        DicomMetadata: Populated DicomMetadata created with OCT metadata
    """
    patient_info = fds_patient_meta(oct.metadata)
    series_info = fds_series_meta(oct.metadata)
    manufacturer_info = fds_manu_meta(oct.metadata, oct.header)
    header = oct.oct_header or {}
    image_geometry = fds_image_geom(oct.pixel_spacing, header.get("scan_mode"))
    oct_image_params = fds_image_params(manufacturer_info.manufacturer_model)
    n_frames = int(header.get("number_slices") or 0) or len(oct.volume or [])
    series_info.scan_pattern = topcon_scan_pattern(header.get("scan_mode"), n_frames)
    series_info.scan_pattern_specified = True

    return DicomMetadata(
        patient_info=patient_info,
        series_info=series_info,
        manufacturer_info=manufacturer_info,
        image_geometry=image_geometry,
        oct_image_params=oct_image_params,
    )
