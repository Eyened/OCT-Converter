from __future__ import annotations

from datetime import datetime

from oct_converter.dicom.metadata import (
    DicomMetadata,
    ImageGeometry,
    ManufacturerMeta,
    OCTDeviceSpecs,
    OCTImageParams,
    OPTAcquisitionDevice,
    OPTAnatomyStructure,
    PatientMeta,
    SeriesMeta,
)
from oct_converter.image_types import OCTVolumeWithMetaData
from oct_converter.readers.scan_geometry import topcon_scan_pattern


def _model_name(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\x00", "").strip()


def _normalize_model_key(value: str) -> str:
    """Lowercase alphanumerics only, for alias matching against HW_INFO strings."""
    return "".join(ch for ch in value.lower() if ch.isalnum())


# Published Topcon brochure values. ``dicom_*`` stay unset until those
# quantities are shown to match PS3.3 C.8.17.9 inherent limiting resolution
# (and bandwidth convention for IlluminationBandwidth).
_TRITON_SPECS = OCTDeviceSpecs(
    wavelength_nm=1050,
    optical_axial_resolution_um=8,
    optical_lateral_resolution_um=20,
)
_OCT_2000_SPECS = OCTDeviceSpecs(
    wavelength_nm=840,
    bandwidth_nm=50,  # brochure "half bandwidth"; not DICOM-validated
    optical_axial_resolution_um=5.5,
    optical_lateral_resolution_um=20,
)
_OCT_1000_SPECS = OCTDeviceSpecs(
    wavelength_nm=840,
    bandwidth_nm=50,  # brochure "half bandwidth"; not DICOM-validated
    optical_axial_resolution_um=5.5,
    optical_lateral_resolution_um=20,
)

# Longer aliases first so "tritonplus" wins over "triton".
_TOPCON_MODEL_ALIASES: tuple[tuple[tuple[str, ...], OCTDeviceSpecs], ...] = (
    (("tritonplus", "triton+"), _TRITON_SPECS),
    (("triton",), _TRITON_SPECS),
    (("3doct2000", "oct2000"), _OCT_2000_SPECS),
    (("3doct1000", "oct1000"), _OCT_1000_SPECS),
)


def lookup_topcon_device_specs(model_name: str | None) -> OCTDeviceSpecs | None:
    """Return published specs for an ``@HW_INFO`` model string, if known."""
    key = _normalize_model_key(_model_name(model_name))
    if not key:
        return None
    for aliases, specs in _TOPCON_MODEL_ALIASES:
        if any(alias in key for alias in aliases):
            return specs
    return None


def apply_topcon_device_specs(image_params: OCTImageParams, model_name: str) -> None:
    """Copy only DICOM-validated device fields from the matched brochure record.

    Wavelength is treated as the nominal illuminator wavelength and is written.
    Bandwidth and spatial-resolution brochure values are retained on
    ``manufacturer_specs`` but are not copied into C.8.17.9 tags until their
    equivalence to the DICOM definitions is established. Illumination power
    and detector type are never invented from brochures here.
    """
    specs = lookup_topcon_device_specs(model_name)
    if specs is None:
        return

    image_params.manufacturer_specs = specs

    if specs.wavelength_nm is not None:
        image_params.IlluminationWaveLength = specs.wavelength_nm
        image_params.device_specs = True

    if specs.dicom_bandwidth_nm is not None:
        image_params.IlluminationBandwidth = specs.dicom_bandwidth_nm
        image_params.device_specs = True

    if specs.dicom_depth_resolution_um is not None:
        image_params.DepthSpatialResolution = specs.dicom_depth_resolution_um
        image_params.device_specs = True
    if specs.dicom_alongscan_resolution_um is not None:
        image_params.AlongscanSpatialResolution = specs.dicom_alongscan_resolution_um
        image_params.device_specs = True
    if specs.dicom_acrossscan_resolution_um is not None:
        image_params.AcrossscanSpatialResolution = specs.dicom_acrossscan_resolution_um
        image_params.device_specs = True


def opt_series_fallback(scan_mode: int | None, octa: bool) -> str:
    """Series description used when the capture label is empty."""
    if octa:
        return "OCTA"
    mode = int(scan_mode) if scan_mode is not None else -1
    return {0: "Line", 1: "Circle", 3: "Radial", 4: "Cross", 11: "Cross"}.get(
        mode, "OCT"
    )


def fda_patient_meta(fda_metadata: dict) -> PatientMeta:
    """Creates PatientMeta from FDA metadata

    Args:
        fda_metadata: Nested dictionary of metadata collected by the fda reader
    Returns:
        PatientMeta: Patient metadata populated by fda_metadata
    """
    patient_info = fda_metadata.get("patient_info_02") or fda_metadata.get(
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


def fda_series_meta(fda_metadata: dict) -> SeriesMeta:
    """Creates SeriesMeta from FDA metadata

    Args:
        fda_metadata: Nested dictionary of metadata collected by the fda reader
    Returns:
        SeriesMeta: Series metadata populated by fda_metadata
    """
    capture_info = fda_metadata.get("capture_info_02") or fda_metadata.get(
        "capture_info", {}
    )
    lat_map = {0: "R", 1: "L", None: ""}
    series = SeriesMeta()

    series.study_id = ""
    series.series_id = capture_info.get("session_id", "")
    series.laterality = lat_map[capture_info.get("eye", None)]
    series.acquisition_date = datetime(*capture_info.get("cap_date"))
    series.opt_anatomy = OPTAnatomyStructure.Retina
    label = str(capture_info.get("label") or "").strip()
    series.protocol = label
    series.description = label

    return series


def fda_manu_meta(fda_metadata: dict, fda_header: dict) -> ManufacturerMeta:
    """Creates ManufacturerMeta from FDA metadata

    Args:
        fda_metadata: Nested dictionary of metadata collected by the fda reader
        fda_header: Dictionary of file header information parsed by the fda reader
    Returns:
        ManufacturerMeta: Manufacture metadata populated by fda_metadata
    """
    hw_info = (
        fda_metadata.get("hw_info_03")
        or fda_metadata.get("hw_info_02")
        or fda_metadata.get("hw_info_01", {})
    )
    manufacture = ManufacturerMeta()

    manufacture.manufacturer = "Topcon"
    manufacture.manufacturer_model = _model_name(hw_info.get("model_name"))
    manufacture.device_serial = _model_name(hw_info.get("serial_number"))
    manufacture.software_version = _software_version(fda_metadata, fda_header)

    return manufacture


def _software_version(fda_metadata: dict, fda_header: dict) -> str:
    """Software string stored in the FDA, not the container version.

    ``@REGIST_INFO`` carries the IMAGEnet/OCT software version. The file
    header's major.minor pair is the FDA container version and is only used
    when no software string is present.
    """
    regist = (fda_metadata.get("regist_info") or {}).get("version")
    if isinstance(regist, str) and regist.strip():
        return regist.strip()

    main = fda_metadata.get("main_module_info") or {}
    numbers = [main.get(f"file_version_{index}") for index in range(1, 5)]
    if any(number is not None for number in numbers):
        version = ".".join(str(int(number)) for number in numbers if number is not None)
        name = str(main.get("software_name") or "").strip()
        return f"{name} {version}".strip() if name else version

    major = fda_header.get("major_ver")
    minor = fda_header.get("minor_ver")
    if major is not None and minor is not None:
        return f"{major}.{minor}"
    return ""


def fda_image_geom(pixel_spacing: list) -> ImageGeometry:
    """Creates ImageGeometry from FDA metadata

    Args:
        pixel_spacing: Pixel spacing calculated in the fda reader
    Returns:
        ImageGeometry: Geometry data populated by pixel_spacing
    """
    image_geom = ImageGeometry()
    image_geom.pixel_spacing = [pixel_spacing[2], pixel_spacing[0]]
    image_geom.slice_thickness = pixel_spacing[1]
    image_geom.image_orientation = [1, 0, 0, 0, 1, 0]

    return image_geom


def fda_image_params(model_name: str) -> OCTImageParams:
    """OCT parameters identified by the model string in ``@HW_INFO``.

    The file is an OCT scan, so the acquisition device is the OCT scanner.
    Only DICOM-validated brochure fields (currently nominal wavelength) are
    written into C.8.17.9 tags; other published optical values stay on
    ``manufacturer_specs``. Unset when the model is missing or unknown.
    """
    image_params = OCTImageParams()
    image_params.opt_acquisition_device = OPTAcquisitionDevice.OCTScanner
    apply_topcon_device_specs(image_params, model_name)
    return image_params


def fda_dicom_metadata(oct: OCTVolumeWithMetaData) -> DicomMetadata:
    """Creates DicomMetadata and populates each module

    Args:
        oct: OCTVolumeWithMetaData created by the fda reader
    Returns:
        DicomMetadata: Populated DicomMetadata created with OCT metadata
    """
    meta = DicomMetadata
    meta.patient_info = fda_patient_meta(oct.metadata)
    meta.series_info = fda_series_meta(oct.metadata)
    meta.manufacturer_info = fda_manu_meta(oct.metadata, oct.header)
    meta.image_geometry = fda_image_geom(oct.pixel_spacing)
    meta.oct_image_params = fda_image_params(meta.manufacturer_info.manufacturer_model)
    header = oct.oct_header or {}
    n_frames = int(header.get("number_slices") or 0) or len(oct.volume or [])
    meta.series_info.scan_pattern = topcon_scan_pattern(header.get("scan_mode"), n_frames)
    meta.series_info.octa = bool((oct.metadata or {}).get("oct_angiography"))

    return meta
