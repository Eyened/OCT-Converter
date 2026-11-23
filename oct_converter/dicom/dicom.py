from __future__ import annotations

import typing as t
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import (
    UID,
    ExplicitVRLittleEndian,
    JPEG2000,
    JPEG2000Lossless,
    OphthalmicOpticalCoherenceTomographyBscanVolumeAnalysisStorage,
    OphthalmicPhotography8BitImageStorage,
    OphthalmicPhotography16BitImageStorage,
    OphthalmicTomographyImageStorage,
    generate_uid,
)
from pydicom.valuerep import DSfloat

from oct_converter.dicom.export import SeriesUidPlan, prepare_export
from oct_converter.dicom.j2k import (
    J2KFrames,
    encode_j2k_lossless_frames,
    prepare_j2k_frames,
)
from oct_converter.dicom.heightmap import (
    HEIGHTMAP_PADDING_VALUE,
    contours_to_heightmaps,
)
from oct_converter.dicom.metadata import (
    DicomMetadata,
    OCTDetectorType,
    OPTAcquisitionDevice,
)
from oct_converter.dicom.parse import (
    read_boct_study,
    read_e2e_study,
    read_fda_study,
    read_fds_study,
    read_img_study,
    read_poct_study,
    read_study,
)
from oct_converter.dicom.write import write_dicom_study
from oct_converter.readers.scan_geometry import (
    ScanGeometry,
    circle_reference_coordinates,
    scan_pattern_code,
)

# Deterministic implementation UID based on package name and version
version = metadata.version("oct_converter")
implementation_uid = generate_uid(entropy_srcs=["oct_converter", version])

# PS3.4 Height Map Segmentation Storage (not yet in all pydicom releases)
HeightMapSegmentationStorage = UID("1.2.840.10008.5.1.4.1.1.66.8")

# Acquisition timestamps from readers are either datetime or this string form.
_ACQUISITION_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _coerce_datetime(value: datetime | str | None) -> datetime | None:
    """Normalize reader acquisition timestamps to ``datetime`` or ``None``."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.strptime(value, _ACQUISITION_DATETIME_FORMAT)
    raise TypeError(f"Expected datetime or str, got {type(value)!r}")


def set_content_date_time(ds: Dataset, when: datetime | None = None) -> None:
    """Set ContentDate / ContentTime from when the pixels were acquired.

    An empty timestamp stays empty. Conversion time is not a substitute.
    """
    if when is None:
        ds.ContentDate = ""
        ds.ContentTime = ""
        return
    ds.ContentDate = when.strftime("%Y%m%d")
    ds.ContentTime = when.strftime("%H%M%S.%f")


def format_dicom_date(value: datetime | str | None) -> str:
    """Format a timestamp as DICOM DA (YYYYMMDD), or ``\"\"`` if missing."""
    dt = _coerce_datetime(value)
    return dt.strftime("%Y%m%d") if dt else ""


def format_dicom_time(value: datetime | str | None) -> str:
    """Format a timestamp as DICOM TM (HHMMSS.ffffff), or ``\"\"`` if missing."""
    dt = _coerce_datetime(value)
    return dt.strftime("%H%M%S.%f") if dt else ""


def format_acquisition_datetime(value: datetime | str | None) -> str:
    """Format a timestamp as DICOM DT, or ``\"\"`` if missing."""
    dt = _coerce_datetime(value)
    return dt.strftime("%Y%m%d%H%M%S.%f") if dt else ""


def opt_base_dicom(filepath: Path) -> Dataset:
    """Creates the base dicom to be populated.

    Args:
            filepath: Path to where output file is to be saved
    Returns:
            ds: FileDataset with file meta, preamble, and empty dataset
    """
    # Populate required values for file meta information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = OphthalmicTomographyImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = implementation_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Create the FileDataset instance with file meta, preamble and empty DS
    ds = FileDataset(str(filepath), {}, file_meta=file_meta, preamble=b"\0" * 128)
    return ds


def populate_patient_info(ds: Dataset, meta: DicomMetadata) -> Dataset:
    """Populates Patient Module PS3.3 C.7.1.1

    Args:
            ds: current dataset
            meta: DICOM metadata information
    Returns:
            ds: Dataset, updated with patient information
    """
    # Patient Module PS3.3 C.7.1.1
    last = meta.patient_info.last_name or ""
    first = meta.patient_info.first_name or ""
    # A lone "^" is a delimiter with no name. Type 2 unknown is zero-length.
    ds.PatientName = f"{last}^{first}" if last or first else ""
    ds.PatientID = meta.patient_info.patient_id or ""
    ds.PatientSex = meta.patient_info.patient_sex or ""
    ds.PatientBirthDate = (
        meta.patient_info.patient_dob.strftime("%Y%m%d")
        if meta.patient_info.patient_dob
        else ""
    )
    return ds


def populate_manufacturer_info(ds: Dataset, meta: DicomMetadata) -> Dataset:
    """Populates equipment modules PS3.3 C.7.5.1, PS3.3 C.7.5.2

    Args:
            ds: current dataset
            meta: DICOM metadata information
    Returns:
            ds: Dataset, updated with equipment information
    """
    # General and enhanced equipment module PS3.3 C.7.5.1, PS3.3 C.7.5.2
    ds.Manufacturer = meta.manufacturer_info.manufacturer
    ds.ManufacturerModelName = meta.manufacturer_info.manufacturer_model
    ds.DeviceSerialNumber = meta.manufacturer_info.device_serial
    ds.SoftwareVersions = meta.manufacturer_info.software_version

    # OPT parameter module PS3.3 C.8.17.9
    cd, cv, cm = meta.oct_image_params.opt_acquisition_device.value
    ds.AcquisitionDeviceTypeCodeSequence = [Dataset()]
    ds.AcquisitionDeviceTypeCodeSequence[0].CodeValue = cv
    ds.AcquisitionDeviceTypeCodeSequence[0].CodingSchemeDesignator = cd
    ds.AcquisitionDeviceTypeCodeSequence[0].CodeMeaning = cm
    # Only a detector reported by the source file. UNKNOWN is not a DICOM
    # enumerated value and is not a measurement.
    if meta.oct_image_params.DetectorType is not OCTDetectorType.Unknown:
        ds.DetectorType = meta.oct_image_params.DetectorType.value
    if meta.oct_image_params.device_specs:
        _apply_device_specs(ds, meta)
    return ds


def _apply_device_specs(ds: Dataset, meta: DicomMetadata) -> None:
    """Write optical tags that were taken from the device specification."""
    params = meta.oct_image_params
    for field, keyword in (
        ("IlluminationWaveLength", "IlluminationWaveLength"),
        ("IlluminationPower", "IlluminationPower"),
        ("IlluminationBandwidth", "IlluminationBandwidth"),
        ("DepthSpatialResolution", "DepthSpatialResolution"),
        ("MaximumDepthDistortion", "MaximumDepthDistortion"),
        ("AlongscanSpatialResolution", "AlongScanSpatialResolution"),
        ("MaximumAlongscanDistortion", "MaximumAlongScanDistortion"),
        ("AcrossscanSpatialResolution", "AcrossScanSpatialResolution"),
        ("MaximumAcrossscanDistortion", "MaximumAcrossScanDistortion"),
    ):
        value = getattr(params, field)
        if value is not None:
            setattr(ds, keyword, value)


def populate_opt_series(
    ds: Dataset,
    meta: DicomMetadata,
    study_instance_uid: str | None = None,
    series_instance_uid: str | None = None,
    sop_instance_uid: str | None = None,
) -> Dataset:
    """Populates study and series modules, PS3.3 C.7.2.1, PS3.3 C.7.3.1,
    PS3.3 C.8.17.6, and PS3.3 C.12.1

    Args:
            ds: current dataset
            meta: DICOM metadata information
            study_instance_uid: Optional shared Study Instance UID
            series_instance_uid: Optional Series Instance UID
            sop_instance_uid: Optional SOP Instance UID
    Returns:
            ds: Dataset, updated with study and series information
    """
    # General study / series UIDs (optional overrides keep OPT/SEG/fundus linked)
    ds.StudyInstanceUID = study_instance_uid or generate_uid()
    ds.SeriesInstanceUID = series_instance_uid or generate_uid()
    ds.StudyID = meta.series_info.study_id
    ds.StudyDate = format_dicom_date(meta.series_info.acquisition_date)
    ds.StudyTime = format_dicom_time(meta.series_info.acquisition_date)
    # General Study Type 2. Present and empty when the source has no value.
    if "ReferringPhysicianName" not in ds:
        ds.ReferringPhysicianName = ""
    if "AccessionNumber" not in ds:
        ds.AccessionNumber = ""
    ds.Laterality = meta.series_info.laterality
    if meta.series_info.protocol:
        ds.ProtocolName = meta.series_info.protocol
    ds.SeriesDescription = meta.series_info.description
    # Ophthalmic Tomography Series PS3.3 C.8.17.6
    ds.Modality = "OPT"
    try:
        ds.SeriesNumber = int(meta.series_info.series_id)
    except (TypeError, ValueError):
        ds.SeriesNumber = 1

    # SOP Common module PS3.3 C.12.1
    ds.SOPClassUID = OphthalmicTomographyImageStorage
    ds.SOPInstanceUID = sop_instance_uid or generate_uid()
    return ds


def populate_ocular_region(ds: Dataset, meta: DicomMetadata) -> Dataset:
    """Populates ocular region modules, PS3.3 C.8.17.5, PS3.3 C.7.6.16.2.8,
    and PS3.3 C.7.6.16.2.1

    Args:
            ds: current dataset
            meta: DICOM metadata information
    Returns:
            ds: Dataset, updated with ocular region information
    """
    # Ocular region imaged module PS3.3 C.8.17.5
    cd, cv, cm = meta.series_info.opt_anatomy.value
    ds.ImageLaterality = meta.series_info.laterality
    ds.AnatomicRegionSequence = [Dataset()]
    ds.AnatomicRegionSequence[0].CodeValue = cv
    ds.AnatomicRegionSequence[0].CodingSchemeDesignator = cd
    ds.AnatomicRegionSequence[0].CodeMeaning = cm
    return ds


def opt_shared_functional_groups(ds: Dataset, meta: DicomMetadata) -> Dataset:
    # ---- Shared
    shared_ds = [Dataset()]
    # Frame anatomy PS3.3 C.7.6.16.2.8
    shared_ds[0].FrameAnatomySequence = [Dataset()]
    shared_ds[0].FrameAnatomySequence[0].FrameLaterality = meta.series_info.laterality
    shared_ds[0].FrameAnatomySequence[0].AnatomicRegionSequence = [
        ds.AnatomicRegionSequence[0].copy()
    ]
    # Pixel Measures PS3.3 C.7.6.16.2.1
    shared_ds[0].PixelMeasuresSequence = [Dataset()]
    shared_ds[0].PixelMeasuresSequence[
        0
    ].PixelSpacing = meta.image_geometry.pixel_spacing
    # A line, circle, or cross has no stack interval. Zero is not a thickness.
    if meta.image_geometry.slice_thickness and meta.image_geometry.slice_thickness > 0:
        shared_ds[0].PixelMeasuresSequence[
            0
        ].SliceThickness = meta.image_geometry.slice_thickness
    # Plane Orientation PS3.3 C.7.6.16.2.4
    shared_ds[0].PlaneOrientationSequence = [Dataset()]
    shared_ds[0].PlaneOrientationSequence[
        0
    ].ImageOrientationPatient = meta.image_geometry.image_orientation
    ds.SharedFunctionalGroupsSequence = shared_ds
    return ds


def _encode_pixel_array(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Store pixels as the smallest valid DICOM integer type.

    Checks for datatype and value limits in order to set the appropriate encoding.
    Empty arrays are rejected.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError("Cannot encode an empty array as DICOM pixel data.")

    if np.issubdtype(arr.dtype, np.integer):
        if np.issubdtype(arr.dtype, np.signedinteger):
            if arr.min() >= -128 and arr.max() <= 127:
                return np.ascontiguousarray(arr.astype(np.int8)), 8
            return np.ascontiguousarray(arr.astype(np.int16)), 16
        if arr.max() <= 255:
            return np.ascontiguousarray(arr, dtype=np.uint8), 8
        return np.ascontiguousarray(arr, dtype=np.uint16), 16

    mx = float(arr.max())
    if mx <= 1.0:
        out = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(out), 8
    if mx <= 255:
        out = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(out), 8
    mn = float(arr.min())
    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint16), 16
    out = ((arr - mn) / (mx - mn) * 65535.0).clip(0, 65535)
    return np.ascontiguousarray(np.rint(out).astype(np.uint16)), 16


def _as_grayscale_array(frames: t.Any) -> np.ndarray:
    """Coerce fundus pixel input to a 2-D grayscale image.

    RGB / RGBA arrays (HWC or CHW) are reduced to luminance so MONOCHROME2
    PixelData length matches Rows x Columns. Plain 2-D arrays are unchanged.
    """
    arr = np.asarray(frames)
    if arr.ndim == 2:
        return arr
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float32)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    elif arr.ndim == 3 and arr.shape[0] in (3, 4):
        rgb = arr[:3].astype(np.float32)
        return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    raise ValueError(
        f"Expected 2-D grayscale or 3-channel fundus image, got shape {arr.shape}"
    )


def _pixel_data_bytes(arr: np.ndarray) -> bytes:
    """Serialize pixels and pad to even length (DICOM PixelData)."""
    raw = np.ascontiguousarray(arr).tobytes()
    if len(raw) % 2:
        raw += b"\x00"
    return raw


def _apply_pixel_encoding_tags(
    ds: Dataset,
    bits: int,
    samples_per_pixel: int = 1,
    photometric: str = "MONOCHROME2",
    pixel_representation: int = 0,
    bits_stored: int | None = None,
    planar: bool | None = None,
) -> None:
    """Set bit-depth, photometric, and full-range VOI window tags."""
    stored = bits if bits_stored is None else bits_stored
    ds.SamplesPerPixel = samples_per_pixel
    ds.PhotometricInterpretation = photometric
    ds.PixelRepresentation = pixel_representation
    ds.BitsAllocated = bits
    ds.BitsStored = stored
    ds.HighBit = stored - 1
    if planar is None:
        planar = photometric == "RGB"
    if planar:
        ds.PlanarConfiguration = 0
    elif "PlanarConfiguration" in ds:
        del ds.PlanarConfiguration
    # PS3.3 C.11.2.1.2.2: windowing applies only to MONOCHROME1 and MONOCHROME2.
    if photometric in ("MONOCHROME1", "MONOCHROME2"):
        window_bits = stored
        ds.WindowCenter = 0 if pixel_representation == 1 else 1 << (window_bits - 1)
        ds.WindowWidth = 1 << window_bits


def _apply_j2k_pixels(ds: Dataset, frames: J2KFrames) -> None:
    """Encapsulate JPEG 2000 codestreams and set the matching transfer syntax.

    The codestreams are the bytes taken from the source file. Nothing is
    decoded or encoded again. A reversible 5-3 wavelet uses the JPEG 2000
    lossless transfer syntax; an irreversible 9-7 wavelet uses the lossy one.
    """
    syntax = JPEG2000Lossless if frames.reversible else JPEG2000
    ds.file_meta.TransferSyntaxUID = syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    _apply_pixel_encoding_tags(
        ds,
        frames.bits_allocated,
        samples_per_pixel=frames.samples,
        photometric=frames.photometric,
        pixel_representation=frames.pixel_representation,
        bits_stored=frames.bits_stored,
        # Compressed samples are not planar. Colour images still carry
        # Planar Configuration 0 so decoders that require the tag accept them.
        planar=frames.samples > 1,
    )
    ds.LossyImageCompression = "00" if frames.reversible else "01"
    if not frames.reversible:
        stored = sum(len(codestream) for codestream in frames.codestreams)
        uncompressed = (
            frames.rows
            * frames.columns
            * frames.samples
            * (frames.bits_allocated // 8)
            * len(frames.codestreams)
        )
        if stored:
            ds.LossyImageCompressionRatio = DSfloat(
                uncompressed / stored, auto_format=True
            )
        ds.LossyImageCompressionMethod = "ISO_15444_1"
    ds.PixelData = encapsulate(list(frames.codestreams))
    pixel_data = ds["PixelData"]
    pixel_data.VR = "OB"
    pixel_data.is_undefined_length = True


def _photography_timing(ds: Dataset, meta: DicomMetadata) -> None:
    """Photography uses image laterality. Content time is when the pixels were acquired."""
    if "Laterality" in ds:
        del ds.Laterality
    acquired = format_dicom_date(meta.series_info.acquisition_date)
    if acquired:
        ds.ContentDate = acquired
        ds.ContentTime = format_dicom_time(meta.series_info.acquisition_date)


def _photography_sop_uid(bits: int) -> UID:
    """Ophthalmic Photography SOP class matching stored bit depth."""
    if bits == 8:
        return OphthalmicPhotography8BitImageStorage
    return OphthalmicPhotography16BitImageStorage


def _ophthalmic_image_descriptors(ds: Dataset, monochrome: bool) -> None:
    """Type 1 descriptors that describe the pixels, not the eye."""
    ds.BurnedInAnnotation = "NO"
    if monochrome:
        ds.PresentationLUTShape = "IDENTITY"


def _add_empty(ds: Dataset, keyword: str) -> None:
    """Insert an empty element using the dictionary VR."""
    if keyword in ds:
        return
    from pydicom.datadict import dictionary_VR, tag_for_keyword

    tag = tag_for_keyword(keyword)
    ds.add_new(tag, dictionary_VR(tag), None)


def _empty_opt_acquisition(ds: Dataset) -> None:
    """Type 2 OPT acquisition attributes. Empty means the FDA did not record them."""
    ds.PositionReferenceIndicator = ""
    ds.AcquisitionContextSequence = []
    # Not a concatenation. Required by the OPT image module.
    ds.ConcatenationFrameOffsetNumber = 0
    ds.InConcatenationNumber = 1
    ds.InConcatenationTotalNumber = 1
    for keyword in (
        "AxialLengthOfTheEye",
        "HorizontalFieldOfView",
        "EmmetropicMagnification",
        "IntraOcularPressure",
        "PupilDilated",
    ):
        _add_empty(ds, keyword)
    ds.RefractiveStateSequence = []
    ds.LightPathFilterTypeStackCodeSequence = []


def _empty_photo_acquisition(ds: Dataset) -> None:
    """Type 2 photography acquisition attributes, present and empty."""
    for keyword in (
        "HorizontalFieldOfView",
        "EmmetropicMagnification",
        "IntraOcularPressure",
        "PupilDilated",
    ):
        _add_empty(ds, keyword)
    ds.RefractiveStateSequence = []


def _opt_multiframe_dimensions(ds: Dataset) -> None:
    """Multi-frame Dimension module: one stack, ordered by B-scan."""
    dim_uid = generate_uid()
    dim_org = Dataset()
    dim_org.DimensionOrganizationUID = dim_uid
    ds.DimensionOrganizationSequence = [dim_org]
    ds.DimensionOrganizationType = "3D"

    def _index(pointer: tuple[int, int]) -> Dataset:
        item = Dataset()
        item.DimensionOrganizationUID = dim_uid
        item.DimensionIndexPointer = pointer
        item.FunctionalGroupPointer = (0x0020, 0x9111)  # Frame Content Sequence
        return item

    ds.DimensionIndexSequence = [
        _index((0x0020, 0x9056)),  # Stack ID
        _index((0x0020, 0x9057)),  # In-Stack Position Number
    ]


def write_oct_bscan_volume_analysis_dicom(
    meta: DicomMetadata,
    frames: t.Sequence[np.ndarray],
    opt_ds: Dataset,
    filepath: Path,
    *,
    bscans_per_frame: int = 1,
    series_instance_uid: str | None = None,
    sop_instance_uid: str | None = None,
) -> FileDataset:
    """Write Ophthalmic OCT B-scan Volume Analysis (flow / OCTA) DICOM.

    Companion to an Ophthalmic Tomography Image. Shares Study and Frame of
    Reference UIDs with ``opt_ds``. Modality is ``OPTBSV`` (PS3.3 C.8.17.18).
    Pixel Representation is signed (IOD requirement); input frames are cast
    to ``int16``.

    Args:
        meta: Shared DICOM metadata (patient, geometry, manufacturer).
        frames: Flow B-scans, one ``(rows, cols)`` array per OPT frame.
        opt_ds: Structural OPT instance this volume was derived from.
        filepath: Output ``.dcm`` path.
        bscans_per_frame: Repeats at each location (``0022,1642``).
        series_instance_uid: Optional Series Instance UID override.
        sop_instance_uid: Optional SOP Instance UID override.
    """
    if not frames:
        raise ValueError("Angiography volume has no frames")
    n_opt = int(getattr(opt_ds, "NumberOfFrames", 1) or 1)
    if len(frames) != n_opt:
        raise ValueError(
            f"Angiography has {len(frames)} frames but OPT has {n_opt}"
        )

    filepath = Path(filepath)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = (
        OphthalmicOpticalCoherenceTomographyBscanVolumeAnalysisStorage
    )
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid or generate_uid()
    file_meta.ImplementationClassUID = implementation_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(filepath), {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds = populate_patient_info(ds, meta)
    # AcquisitionDeviceTypeCodeSequence is OPT-only; strip after equipment tags.
    ds = populate_manufacturer_info(ds, meta)
    if "AcquisitionDeviceTypeCodeSequence" in ds:
        del ds.AcquisitionDeviceTypeCodeSequence

    # Study / Series — OPTBSV is a separate series from the structural OPT.
    ds.StudyInstanceUID = opt_ds.StudyInstanceUID
    ds.SeriesInstanceUID = series_instance_uid or generate_uid()
    ds.StudyID = meta.series_info.study_id
    ds.StudyDate = format_dicom_date(meta.series_info.acquisition_date)
    ds.StudyTime = format_dicom_time(meta.series_info.acquisition_date)
    ds.ReferringPhysicianName = getattr(opt_ds, "ReferringPhysicianName", "") or ""
    ds.AccessionNumber = getattr(opt_ds, "AccessionNumber", "") or ""
    ds.Laterality = meta.series_info.laterality
    ds.SeriesDescription = meta.series_info.description or "OCTA flow"
    ds.Modality = "OPTBSV"
    try:
        ds.SeriesNumber = int(meta.series_info.series_id) + 1
    except (TypeError, ValueError):
        ds.SeriesNumber = int(getattr(opt_ds, "SeriesNumber", 1) or 1) + 1

    ds.SOPClassUID = OphthalmicOpticalCoherenceTomographyBscanVolumeAnalysisStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.FrameOfReferenceUID = opt_ds.FrameOfReferenceUID

    ds = populate_ocular_region(ds, meta)

    # Image module — signed pixels, monochrome flow volume.
    ds.ImageType = ["ORIGINAL", "PRIMARY"]
    set_content_date_time(ds, _coerce_datetime(meta.series_info.acquisition_date))
    ds.InstanceNumber = 1
    ds.BurnedInAnnotation = "NO"
    ds.RecognizableVisualFeatures = "NO"
    ds.PresentationLUTShape = "IDENTITY"
    ds.LossyImageCompression = "00"
    ds.ConcatenationFrameOffsetNumber = 0
    ds.InConcatenationNumber = 1
    ds.InConcatenationTotalNumber = 1

    signed = [np.ascontiguousarray(np.asarray(f), dtype=np.int16) for f in frames]
    j2k = prepare_j2k_frames(encode_j2k_lossless_frames(signed))
    if j2k.samples != 1:
        raise ValueError(
            "OCTA flow JPEG 2000 frames must be single component, "
            f"got {j2k.samples}."
        )
    # IOD requires signed samples; the reversible codestream must match.
    if j2k.pixel_representation != 1:
        raise ValueError(
            "OCTA flow JPEG 2000 must be signed (Pixel Representation 1), "
            f"got {j2k.pixel_representation}."
        )
    _apply_j2k_pixels(ds, j2k)
    # Supplement 197 / PS3.3 A.84.3.1: VOI LUT Module must not be used on this
    # SOP Class. Windowing lives in Frame VOI LUT With LUT instead.
    for keyword in ("WindowCenter", "WindowWidth", "WindowCenterWidthExplanation"):
        if keyword in ds:
            del ds[keyword]
    n_frames = len(j2k.codestreams)
    ds.NumberOfFrames = n_frames
    ds.Rows = j2k.rows
    ds.Columns = j2k.columns

    # Acquisition method — Topcon OCTARA ≈ one-sided ratio (CID 4270).
    algo = Dataset()
    algo.AlgorithmFamilyCodeSequence = [
        _code_dataset("DCM", "128304", "OCT-A one-sided ratio (lesser)")
    ]
    algo.AlgorithmName = "OCTARA"
    algo.AlgorithmVersion = str(meta.manufacturer_info.software_version or "unknown")
    ds.AcquisitionMethodAlgorithmSequence = [algo]

    if meta.series_info.scan_pattern:
        scheme, value, meaning = meta.series_info.scan_pattern
    else:
        scheme, value, meaning = scan_pattern_code(None, n_frames)
    slab = float(meta.image_geometry.slice_thickness or 0)
    acq = Dataset()
    acq.ScanPatternTypeCodeSequence = [_code_dataset(scheme, value, meaning)]
    acq.NumberOfBscansPerFrame = max(int(bscans_per_frame), 1)
    acq.BscanSlabThickness = DSfloat(slab, auto_format=True)
    acq.DistanceBetweenBscanSlabs = DSfloat(slab, auto_format=True)
    acq.BscanCycleTime = DSfloat(0, auto_format=True)
    ds.OCTBscanAnalysisAcquisitionParametersSequence = [acq]

    _opt_multiframe_dimensions(ds)

    # Shared functional groups: anatomy, pixel measures, orientation, VOI LUT.
    shared = Dataset()
    shared.FrameAnatomySequence = [Dataset()]
    shared.FrameAnatomySequence[0].FrameLaterality = meta.series_info.laterality
    shared.FrameAnatomySequence[0].AnatomicRegionSequence = [
        ds.AnatomicRegionSequence[0].copy()
    ]
    shared.PixelMeasuresSequence = [Dataset()]
    shared.PixelMeasuresSequence[0].PixelSpacing = meta.image_geometry.pixel_spacing
    if slab > 0:
        shared.PixelMeasuresSequence[0].SliceThickness = meta.image_geometry.slice_thickness
    shared.PlaneOrientationSequence = [Dataset()]
    shared.PlaneOrientationSequence[0].ImageOrientationPatient = (
        meta.image_geometry.image_orientation
    )
    voi = Dataset()
    voi.WindowCenter = 0
    voi.WindowWidth = 1 << 16
    shared.FrameVOILUTSequence = [voi]
    ds.SharedFunctionalGroupsSequence = [shared]

    purpose_struct = _code_dataset(
        "DCM", "128250", "Structural image for imaging processing"
    )
    derivation_code = _code_dataset("DCM", "128303", "OCT B-scan analysis")

    per_frame = []
    for i in range(n_frames):
        frame_fgs = Dataset()
        frame_fgs.PlanePositionSequence = [Dataset()]
        frame_fgs.PlanePositionSequence[0].ImagePositionPatient = [
            0,
            0,
            DSfloat(i * (meta.image_geometry.slice_thickness or 0), auto_format=True),
        ]
        frame_fgs.FrameContentSequence = [Dataset()]
        frame_fgs.FrameContentSequence[0].InStackPositionNumber = i + 1
        frame_fgs.FrameContentSequence[0].StackID = "1"
        frame_fgs.FrameContentSequence[0].DimensionIndexValues = [1, i + 1]

        source = Dataset()
        source.ReferencedSOPClassUID = str(opt_ds.SOPClassUID)
        source.ReferencedSOPInstanceUID = opt_ds.SOPInstanceUID
        source.ReferencedFrameNumber = [i + 1]
        source.PurposeOfReferenceCodeSequence = [purpose_struct]
        source.SpatialLocationsPreserved = "YES"

        deriv = Dataset()
        deriv.DerivationCodeSequence = [derivation_code]
        deriv.SourceImageSequence = [source]
        frame_fgs.DerivationImageSequence = [deriv]

        ref = Dataset()
        ref.ReferencedSOPClassUID = str(opt_ds.SOPClassUID)
        ref.ReferencedSOPInstanceUID = opt_ds.SOPInstanceUID
        ref.ReferencedFrameNumber = [i + 1]
        ref.PurposeOfReferenceCodeSequence = [purpose_struct]
        frame_fgs.ReferencedImageSequence = [ref]

        per_frame.append(frame_fgs)
    ds.PerFrameFunctionalGroupsSequence = per_frame

    ds.save_as(
        filepath, implicit_vr=False, little_endian=True, enforce_file_format=True
    )
    return ds


def write_opt_dicom(
    meta: DicomMetadata,
    frames: t.Sequence[np.ndarray] | None,
    filepath: Path,
    study_instance_uid: str | None = None,
    frame_of_reference_uid: str | None = None,
    series_instance_uid: str | None = None,
    sop_instance_uid: str | None = None,
    fundus_ds: Dataset | None = None,
    compressed_frames: t.Sequence[bytes] | None = None,
) -> FileDataset:
    """Writes required DICOM metadata and oct pixel data to .dcm file.

    Args:
            meta: DICOM metadata information
            frames: list of frames of pixel data
            filepath: Path to where output file is being saved
            study_instance_uid: Optional shared Study Instance UID
            frame_of_reference_uid: Optional shared Frame of Reference UID
            series_instance_uid: Optional Series Instance UID
            sop_instance_uid: Optional SOP Instance UID
            fundus_ds: Optional companion fundus FileDataset for Ophthalmic
                Frame Location (circular / linear scan overlay on localizer)
            compressed_frames: JPEG 2000 frame payloads to encapsulate in
                place of ``frames``. ``frames`` is ignored when this is set.
    Returns:
            FileDataset of the written OPT instance (path is ``filepath``)
    """
    ds = opt_base_dicom(filepath)
    ds = populate_patient_info(ds, meta)
    ds = populate_manufacturer_info(ds, meta)
    ds = populate_opt_series(
        ds,
        meta,
        study_instance_uid=study_instance_uid,
        series_instance_uid=series_instance_uid,
        sop_instance_uid=sop_instance_uid,
    )
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds = populate_ocular_region(ds, meta)
    ds = opt_shared_functional_groups(ds, meta)

    # Frame of Reference Module PS3.3 C.7.4.1
    ds.FrameOfReferenceUID = frame_of_reference_uid or generate_uid()

    # OPT Image Module PS3.3 C.8.17.7
    ds.ImageType = ["ORIGINAL", "PRIMARY"]
    # Value 3 distinguishes an angiography acquisition from a structural cube.
    # Both are still a cube in CID 4272; the flow volume is a separate object
    # (Supplement 197 B-scan Volume Analysis, SOP …77.1.5.8).
    if meta.series_info.octa:
        ds.ImageType.append("ANGIO")
    ds.AcquisitionDateTime = format_acquisition_datetime(
        meta.series_info.acquisition_date
    )
    ds.AcquisitionNumber = 1
    # Supplement 197: YES when FoR / plane macros make the OPT usable as the
    # source for B-scan Volume Analysis or surface segmentation.
    ds.OphthalmicVolumetricPropertiesFlag = "YES"
    _ophthalmic_image_descriptors(ds, monochrome=True)
    _empty_opt_acquisition(ds)
    _opt_multiframe_dimensions(ds)

    j2k = prepare_j2k_frames(list(compressed_frames)) if compressed_frames else None
    if j2k is None and frames is not None:
        # Decoded pixels (e.g. Triton PNG B-scans): re-encode JPEG 2000 lossless.
        pixel_data, bits = _encode_pixel_array(frames)
        if pixel_data.ndim == 2:
            pixel_data = pixel_data[np.newaxis, ...]
        if pixel_data.ndim != 3:
            raise ValueError(
                f"Expected OCT frames as a 3-D volume, got shape {pixel_data.shape}"
            )
        slice_list = [pixel_data[i] for i in range(pixel_data.shape[0])]
        j2k = prepare_j2k_frames(encode_j2k_lossless_frames(slice_list))
    if j2k is not None:
        if j2k.samples != 1:
            raise ValueError(
                "Ophthalmic tomography JPEG 2000 frames must be single component, "
                f"got {j2k.samples}."
            )
        n_frames = len(j2k.codestreams)
        rows = j2k.rows
        columns = j2k.columns
        _apply_j2k_pixels(ds, j2k)
    else:
        raise ValueError(
            "write_opt_dicom requires frames or compressed_frames"
        )
    ds.NumberOfFrames = n_frames
    ds.Rows = rows
    ds.Columns = columns

    # Multi-frame Functional Groups Module PS3.3 C.7.6.16
    # Content time is the acquisition of the B-scans, not when we wrote the file.
    set_content_date_time(ds, _coerce_datetime(meta.series_info.acquisition_date))
    ds.InstanceNumber = 1

    # Scan Pattern Type (CID 4272) — Ophthalmic Tomography Parameters
    geom = meta.scan_geometry
    if meta.series_info.scan_pattern:
        scheme, value, meaning = meta.series_info.scan_pattern
    else:
        scheme, value, meaning = scan_pattern_code(
            (
                ScanGeometry(
                    scan_type=geom.scan_type if geom else "volume",
                    start_angle=geom.start_angle if geom else None,
                    centre=tuple(geom.centre) if geom and geom.centre else None,
                    radius=geom.radius if geom else None,
                )
                if geom
                else None
            ),
            n_frames,
        )
    ds.ScanPatternTypeCodeSequence = [_code_dataset(scheme, value, meaning)]

    per_frame = []
    for i in range(n_frames):
        # Per Frame Functional Groups
        frame_fgs = Dataset()
        frame_fgs.PlanePositionSequence = [Dataset()]
        ipp = [0, 0, DSfloat(i * meta.image_geometry.slice_thickness, auto_format=True)]
        frame_fgs.PlanePositionSequence[0].ImagePositionPatient = ipp
        frame_fgs.FrameContentSequence = [Dataset()]
        frame_fgs.FrameContentSequence[0].InStackPositionNumber = i + 1
        frame_fgs.FrameContentSequence[0].StackID = "1"
        # Dimension Index Sequence order: Stack ID, then In-Stack Position.
        frame_fgs.FrameContentSequence[0].DimensionIndexValues = [1, i + 1]

        # Ophthalmic Frame Location Macro (circular / linear vs fundus)
        if fundus_ds is not None and geom is not None:
            loc = Dataset()
            loc.ReferencedSOPClassUID = fundus_ds.SOPClassUID
            loc.ReferencedSOPInstanceUID = fundus_ds.SOPInstanceUID
            loc.PurposeOfReferenceCodeSequence = [
                _code_dataset("DCM", "121311", "Localizer")
            ]
            if geom.scan_type == "circular" and geom.centre and geom.radius is not None:
                loc.OphthalmicImageOrientation = "NONLINEAR"
                loc.ReferenceCoordinates = circle_reference_coordinates(
                    tuple(geom.centre),
                    float(geom.radius),
                    float(geom.start_angle or 0.0),
                    int(ds.Columns),
                )
                frame_fgs.OphthalmicFrameLocationSequence = [loc]
            else:
                line_start = geom.line_start
                line_end = geom.line_end
                if geom.frame_lines and i < len(geom.frame_lines):
                    fl = geom.frame_lines[i]
                    if fl and fl.get("line_start") and fl.get("line_end"):
                        line_start = fl["line_start"]
                        line_end = fl["line_end"]
                if line_start and line_end:
                    loc.OphthalmicImageOrientation = "LINEAR"
                    # DICOM (row, col) = (y, x)
                    sx, sy = line_start
                    ex, ey = line_end
                    loc.ReferenceCoordinates = [
                        float(sy),
                        float(sx),
                        float(ey),
                        float(ex),
                    ]
                    frame_fgs.OphthalmicFrameLocationSequence = [loc]

        per_frame.append(frame_fgs)
    ds.PerFrameFunctionalGroupsSequence = per_frame
    ds.save_as(
        filepath, implicit_vr=False, little_endian=True, enforce_file_format=True
    )
    return ds


def _code_dataset(scheme: str, value: str, meaning: str) -> Dataset:
    """Build a DICOM coded-entry sequence item."""
    item = Dataset()
    item.CodeValue = value
    item.CodingSchemeDesignator = scheme
    item.CodeMeaning = meaning
    return item


def write_heightmap_seg_dicom(
    meta: DicomMetadata,
    contours: dict,
    opt_ds: Dataset,
    filepath: Path,
    axial_spacing_mm: float | None = None,
    series_instance_uid: str | None = None,
) -> Path | None:
    """Write a Height Map Segmentation DICOM from OCT layer contours.

    Accepts E2E (``contour{id}`` lists) or FDA (named 2-D arrays) contour
    dicts; see ``contours_to_heightmaps``.

    Args:
            meta: DICOM metadata (patient/equipment/geometry)
            contours: ``OCTVolumeWithMetaData.contours`` dict
            opt_ds: Companion OPT FileDataset (must include FOR and SOP UIDs)
            filepath: Output path for the SEG file
            axial_spacing_mm: Axial (B-scan row) spacing in mm for RWVM.
                Defaults to OPT pixel spacing row component when available.
            series_instance_uid: Optional Series Instance UID.
    Returns:
            Path to written SEG file, or None if no contours to encode.
    """
    num_bscans = int(opt_ds.NumberOfFrames)
    width = int(opt_ds.Columns)
    opt_rows = int(opt_ds.Rows)
    layers = contours_to_heightmaps(contours, num_bscans, width)
    if not layers:
        return None

    # Prefer caller-supplied axial spacing; fall back to OPT row spacing.
    if axial_spacing_mm is None:
        if meta.image_geometry.pixel_spacing:
            axial_spacing_mm = float(meta.image_geometry.pixel_spacing[0])
        else:
            axial_spacing_mm = 1.0

    # Heightmap Pixel Measures: row = B-scan spacing, col = OPT column spacing
    opt_col_spacing = (
        float(meta.image_geometry.pixel_spacing[1])
        if meta.image_geometry.pixel_spacing
        else 1.0
    )
    slice_thickness = float(meta.image_geometry.slice_thickness)
    image_orientation = list(meta.image_geometry.image_orientation) or [
        1,
        0,
        0,
        0,
        1,
        0,
    ]

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = HeightMapSegmentationStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = implementation_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(filepath), {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds = populate_patient_info(ds, meta)
    # Equipment without OPT-only AcquisitionDeviceType (still useful)
    ds.Manufacturer = meta.manufacturer_info.manufacturer
    ds.ManufacturerModelName = meta.manufacturer_info.manufacturer_model
    ds.DeviceSerialNumber = meta.manufacturer_info.device_serial
    ds.SoftwareVersions = meta.manufacturer_info.software_version

    # Study / Series / SOP. Study-level attributes match the OPT instance.
    ds.StudyInstanceUID = opt_ds.StudyInstanceUID
    for keyword in (
        "StudyDate",
        "StudyTime",
        "StudyID",
        "AccessionNumber",
        "ReferringPhysicianName",
    ):
        if keyword in opt_ds:
            setattr(ds, keyword, opt_ds[keyword].value)
    if "ReferringPhysicianName" not in ds:
        ds.ReferringPhysicianName = ""
    if "AccessionNumber" not in ds:
        ds.AccessionNumber = ""
    ds.PositionReferenceIndicator = ""
    ds.SeriesInstanceUID = series_instance_uid or generate_uid()
    ds.FrameOfReferenceUID = opt_ds.FrameOfReferenceUID
    ds.Modality = "SEG"
    try:
        ds.SeriesNumber = int(meta.series_info.series_id) + 1000
    except (TypeError, ValueError):
        ds.SeriesNumber = 1000
    ds.Laterality = meta.series_info.laterality
    ds.SeriesDescription = (
        f"{meta.series_info.description} Height Map Segmentation"
        if meta.series_info.description
        else "Height Map Segmentation"
    )
    ds.SOPClassUID = HeightMapSegmentationStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    # General Image / Height Map Segmentation Image Module
    ds.ImageType = ["DERIVED", "PRIMARY"]
    ds.ContentLabel = "HEIGHTMAP"
    ds.ContentDescription = "Retinal layer height map segmentation"
    ds.ContentCreatorName = "oct_converter"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SegmentationType = "HEIGHTMAP"
    ds.NumberOfFrames = len(layers)
    ds.Rows = num_bscans
    ds.Columns = width
    ds.InstanceNumber = 1

    # Floating Point Image Pixel Module PS3.3 C.7.6.24
    ds.BitsAllocated = 32
    ds.FloatPixelPaddingValue = HEIGHTMAP_PADDING_VALUE
    # Keep padding range a single value outside [0, OPT.Rows]
    ds.FloatPixelPaddingRangeLimit = HEIGHTMAP_PADDING_VALUE

    set_content_date_time(ds, _coerce_datetime(meta.series_info.acquisition_date))
    ds.AcquisitionDateTime = format_acquisition_datetime(
        meta.series_info.acquisition_date
    )

    # Segment Sequence
    segment_seq = []
    algorithm_name = (
        meta.manufacturer_info.manufacturer_model
        or meta.manufacturer_info.manufacturer
        or "oct_converter"
    )
    for idx, layer in enumerate(layers, start=1):
        seg = Dataset()
        seg.SegmentNumber = idx
        seg.SegmentLabel = layer.code.label
        seg.SegmentAlgorithmType = "AUTOMATIC"
        seg.SegmentAlgorithmName = str(algorithm_name)
        algo_id = Dataset()
        algo_id.AlgorithmName = str(algorithm_name)
        algo_id.AlgorithmVersion = version
        algo_id.AlgorithmFamilyCodeSequence = [
            _code_dataset("DCM", "123103", "Edge Detection")
        ]
        seg.SegmentationAlgorithmIdentificationSequence = [algo_id]
        seg.SegmentedPropertyCategoryCodeSequence = [
            _code_dataset(
                layer.code.category_scheme,
                layer.code.category_value,
                layer.code.category_meaning,
            )
        ]
        seg.SegmentedPropertyTypeCodeSequence = [
            _code_dataset(
                layer.code.type_scheme,
                layer.code.type_value,
                layer.code.type_meaning,
            )
        ]
        segment_seq.append(seg)
    ds.SegmentSequence = segment_seq

    # Dimension organization (minimal)
    dim_uid = generate_uid()
    dim_org = Dataset()
    dim_org.DimensionOrganizationUID = dim_uid
    ds.DimensionOrganizationSequence = [dim_org]
    dim_index = Dataset()
    dim_index.DimensionOrganizationUID = dim_uid
    dim_index.DimensionIndexPointer = (0x0062, 0x000B)  # Referenced Segment Number
    dim_index.FunctionalGroupPointer = (0x0062, 0x000A)  # Segment Identification Seq
    ds.DimensionIndexSequence = [dim_index]

    # Shared functional groups: Pixel Measures, Derivation Image, RWVM, Plane Orientation
    shared = Dataset()

    if slice_thickness > 0 and opt_col_spacing > 0:
        pm = Dataset()
        pm.PixelSpacing = [slice_thickness, opt_col_spacing]
        shared.PixelMeasuresSequence = [pm]

    if num_bscans > 1:
        po = Dataset()
        # Heightmap plane is orthogonal to OPT B-scans: rows along slice, cols along OPT cols
        # OPT orientation is [row_dir, col_dir]; heightmap row ~ OPT slice (Z), col ~ OPT col
        shared.PlaneOrientationSequence = [Dataset()]
        # Approximate: row direction along patient Z of OPT stack, col along OPT column
        shared.PlaneOrientationSequence[0].ImageOrientationPatient = [
            0,
            0,
            1,
            image_orientation[3],
            image_orientation[4],
            image_orientation[5],
        ]

    # Derivation Image Functional Group
    purpose = _code_dataset(
        "DCM", "121322", "Source Image for Image Processing Operation"
    )
    derivation_code = _code_dataset("DCM", "113076", "Segmentation")
    source = Dataset()
    source.ReferencedSOPClassUID = opt_ds.SOPClassUID
    source.ReferencedSOPInstanceUID = opt_ds.SOPInstanceUID
    source.PurposeOfReferenceCodeSequence = [purpose]
    deriv_item = Dataset()
    deriv_item.DerivationCodeSequence = [derivation_code]
    deriv_item.SourceImageSequence = [source]
    shared.DerivationImageSequence = [deriv_item]

    # Real World Value Mapping: pixel height -> mm
    rwvm = Dataset()
    rwvm.DoubleFloatRealWorldValueFirstValueMapped = 0.0
    rwvm.DoubleFloatRealWorldValueLastValueMapped = float(max(opt_rows, 1))
    rwvm.RealWorldValueIntercept = 0.0
    rwvm.RealWorldValueSlope = float(axial_spacing_mm)
    rwvm.LUTExplanation = "Axial distance from top of B-scan"
    measurement = Dataset()
    measurement.CodeValue = "mm"
    measurement.CodingSchemeDesignator = "UCUM"
    measurement.CodeMeaning = "mm"
    rwvm.MeasurementUnitsCodeSequence = [measurement]
    shared.RealWorldValueMappingSequence = [rwvm]

    ds.SharedFunctionalGroupsSequence = [shared]

    # Per-frame functional groups + float pixel data
    per_frame = []
    float_frames = []
    for idx, layer in enumerate(layers, start=1):
        fg = Dataset()
        # Segment Identification
        seg_id = Dataset()
        seg_id.ReferencedSegmentNumber = idx
        fg.SegmentIdentificationSequence = [seg_id]
        # Frame Content
        fc = Dataset()
        fc.DimensionIndexValues = [idx]
        fc.InStackPositionNumber = idx
        fc.StackID = "1"
        fg.FrameContentSequence = [fc]
        if num_bscans > 1:
            pp = Dataset()
            pp.ImagePositionPatient = [0, 0, 0]
            fg.PlanePositionSequence = [pp]
        per_frame.append(fg)
        float_frames.append(layer.data.astype(np.float32))

    ds.PerFrameFunctionalGroupsSequence = per_frame

    # Common Instance Reference: referenced OPT series
    ref_inst = Dataset()
    ref_inst.ReferencedSOPClassUID = opt_ds.SOPClassUID
    ref_inst.ReferencedSOPInstanceUID = opt_ds.SOPInstanceUID
    ref_series = Dataset()
    ref_series.SeriesInstanceUID = opt_ds.SeriesInstanceUID
    ref_series.ReferencedInstanceSequence = [ref_inst]
    ds.ReferencedSeriesSequence = [ref_series]

    pixel_stack = np.stack(float_frames, axis=0)
    ds.FloatPixelData = pixel_stack.astype("<f4").tobytes()
    ds.save_as(
        filepath, implicit_vr=False, little_endian=True, enforce_file_format=True
    )
    return filepath


def write_fundus_dicom(
    meta: DicomMetadata,
    frames: t.Sequence[np.ndarray] | None,
    filepath: Path,
    study_instance_uid: str | None = None,
    series_instance_uid: str | None = None,
    sop_instance_uid: str | None = None,
    compressed_frames: t.Sequence[bytes] | None = None,
) -> FileDataset:
    """Writes required DICOM metadata and fundus pixel data to .dcm file.

    Args:
            meta: DICOM metadata information
            frames: list of frames of pixel data
            filepath: Path to where output file is being saved
            study_instance_uid: Optional shared Study Instance UID
            series_instance_uid: Optional Series Instance UID
            sop_instance_uid: Optional SOP Instance UID
            compressed_frames: JPEG 2000 payloads to encapsulate. ``frames``
                is ignored when this is set. Single-component images only.
    Returns:
            FileDataset of the written fundus instance
    """
    ds = opt_base_dicom(filepath)
    ds = populate_patient_info(ds, meta)
    ds = populate_manufacturer_info(ds, meta)
    ds = populate_opt_series(
        ds,
        meta,
        study_instance_uid=study_instance_uid,
        series_instance_uid=series_instance_uid,
        sop_instance_uid=sop_instance_uid,
    )
    ds.Modality = "OP"
    j2k = prepare_j2k_frames(list(compressed_frames)) if compressed_frames else None
    if j2k is None and frames is not None:
        gray = _as_grayscale_array(frames)
        pixel_data, _bits = _encode_pixel_array(gray)
        if pixel_data.ndim != 2:
            raise ValueError(
                f"Expected 2-D grayscale fundus image, got shape {pixel_data.shape}"
            )
        j2k = prepare_j2k_frames(encode_j2k_lossless_frames([pixel_data]))
    if j2k is not None:
        if j2k.samples != 1:
            raise ValueError(
                "Grayscale fundus JPEG 2000 must be single component, "
                f"got {j2k.samples}."
            )
        bits = j2k.bits_allocated
        rows = j2k.rows
        columns = j2k.columns
    else:
        raise ValueError("write_fundus_dicom requires frames or compressed_frames")
    photo_sop = _photography_sop_uid(bits)
    ds.SOPClassUID = photo_sop
    ds.file_meta.MediaStorageSOPClassUID = photo_sop
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds = populate_ocular_region(ds, meta)

    _set_image_plane(ds, meta)

    # Ophthalmic photography image. Pixels are the acquired localizer.
    ds.ImageType = ["ORIGINAL", "PRIMARY"]
    enface_to_type = {
        "IR": "RED",
        "FA": "BLUE",
        "ICGA": "GREEN",
    }
    if getattr(ds, "ProtocolName", None) in enface_to_type:
        ds.ImageType.append(enface_to_type.get(ds.ProtocolName))
    ds.AcquisitionDateTime = format_acquisition_datetime(
        meta.series_info.acquisition_date
    )
    _ophthalmic_image_descriptors(ds, monochrome=True)
    _empty_photo_acquisition(ds)
    _apply_j2k_pixels(ds, j2k)
    ds.NumberOfFrames = 1

    # Multi-frame Functional Groups Module PS3.3 C.7.6.16
    _photography_timing(ds, meta)
    ds.InstanceNumber = 1
    ds.Rows = rows
    ds.Columns = columns

    ds.save_as(
        filepath, implicit_vr=False, little_endian=True, enforce_file_format=True
    )
    return ds


def write_color_fundus_dicom(
    meta: DicomMetadata,
    frames: t.Sequence[np.ndarray] | None,
    filepath: Path,
    study_instance_uid: str | None = None,
    series_instance_uid: str | None = None,
    compressed_frames: t.Sequence[bytes] | None = None,
) -> Path:
    """Writes required DICOM metadata and RGB fundus pixel data to .dcm file.

    Args:
            meta: DICOM metadata information
            frames: list of frames of pixel data
            filepath: Path to where output file is being saved
            study_instance_uid: Optional shared Study Instance UID
            series_instance_uid: Optional Series Instance UID
            compressed_frames: JPEG 2000 payloads to encapsulate. ``frames``
                is ignored when this is set. Three-component images only.
    Returns:
            FileDataset of the written fundus instance (path is ``filepath``)
    """
    ds = opt_base_dicom(filepath)
    ds = populate_patient_info(ds, meta)
    ds = populate_manufacturer_info(ds, meta)
    ds = populate_opt_series(
        ds,
        meta,
        study_instance_uid=study_instance_uid,
        series_instance_uid=series_instance_uid,
    )
    ds.Modality = "OP"
    j2k = prepare_j2k_frames(list(compressed_frames)) if compressed_frames else None
    if j2k is None and frames is not None:
        # Topcon colour FDA is BGR-labeled JP2; callers pass corrected RGB
        # pixels. Re-encode JPEG 2000 lossless (no MCT → photometric RGB).
        rgb = np.asarray(frames)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        pixel_data, _bits = _encode_pixel_array(rgb)
        if pixel_data.ndim != 3 or pixel_data.shape[-1] != 3:
            raise ValueError(
                f"Expected HxWx3 RGB fundus image, got shape {pixel_data.shape}"
            )
        j2k = prepare_j2k_frames(encode_j2k_lossless_frames([pixel_data]))
    if j2k is not None:
        if j2k.samples != 3:
            raise ValueError(
                "Colour fundus JPEG 2000 must have 3 components, "
                f"got {j2k.samples}."
            )
        bits = j2k.bits_allocated
        rows = j2k.rows
        columns = j2k.columns
    else:
        raise ValueError(
            "write_color_fundus_dicom requires frames or compressed_frames"
        )
    photo_sop = _photography_sop_uid(bits)
    ds.SOPClassUID = photo_sop
    ds.file_meta.MediaStorageSOPClassUID = photo_sop
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds = populate_ocular_region(ds, meta)

    _set_image_plane(ds, meta)

    # Ophthalmic photography image. Pixels are the acquired fundus.
    ds.ImageType = ["ORIGINAL", "PRIMARY"]
    enface_to_type = {
        "IR": "RED",
        "FA": "BLUE",
        "ICGA": "GREEN",
    }
    if getattr(ds, "ProtocolName", None) in enface_to_type:
        ds.ImageType.append(enface_to_type.get(ds.ProtocolName))
    ds.AcquisitionDateTime = format_acquisition_datetime(
        meta.series_info.acquisition_date
    )
    _ophthalmic_image_descriptors(ds, monochrome=False)
    _empty_photo_acquisition(ds)
    _apply_j2k_pixels(ds, j2k)
    ds.NumberOfFrames = 1

    _photography_timing(ds, meta)
    ds.InstanceNumber = 1
    ds.Rows = rows
    ds.Columns = columns

    ds.save_as(
        filepath, implicit_vr=False, little_endian=True, enforce_file_format=True
    )
    return ds


def _set_image_plane(ds: Dataset, meta: DicomMetadata) -> None:
    """Write pixel spacing and orientation only when the source measured them."""
    spacing = meta.image_geometry.pixel_spacing
    if spacing:
        ds.PixelSpacing = spacing
    orientation = meta.image_geometry.image_orientation
    if orientation:
        ds.ImageOrientationPatient = orientation


def _without_opt_image_plane(meta: DicomMetadata, write):
    """Fundus photos do not inherit the OCT volume's millimetre geometry."""
    saved_spacing = list(meta.image_geometry.pixel_spacing or [])
    saved_orientation = list(meta.image_geometry.image_orientation or [])
    meta.image_geometry.pixel_spacing = []
    meta.image_geometry.image_orientation = []
    try:
        return write()
    finally:
        meta.image_geometry.pixel_spacing = saved_spacing
        meta.image_geometry.image_orientation = saved_orientation


def _with_description(meta: DicomMetadata, fallback: str, write):
    """Use the series description, the capture label, or a name for this object."""
    saved = meta.series_info.description
    label = (saved or meta.series_info.protocol or "").strip()
    meta.series_info.description = label or fallback
    try:
        return write()
    finally:
        meta.series_info.description = saved


def _with_acquisition_device(meta: DicomMetadata, device: OPTAcquisitionDevice, write):
    saved = meta.oct_image_params.opt_acquisition_device
    meta.oct_image_params.opt_acquisition_device = device
    try:
        return write()
    finally:
        meta.oct_image_params.opt_acquisition_device = saved


def _begin_export(
    built: DicomMetadata,
    supplied: DicomMetadata | None,
    overrides: t.Mapping[str, t.Any] | None,
    study_instance_uid: str | None = None,
) -> tuple[DicomMetadata, str, SeriesUidPlan]:
    """Copy parsed metadata, apply overrides, and choose the shared study UID.

    ``supplied`` replaces the metadata built from the file. Overrides are
    applied to a copy of whichever object is used. One study UID is shared
    by every DICOM written from a single input file.
    """
    prepared = prepare_export(supplied if supplied is not None else built, overrides)
    study_uid = prepared.study_instance_uid or study_instance_uid or generate_uid()
    return prepared.meta, study_uid, prepared.series


def create_dicom_from_oct(
    input_file: str,
    output_dir: str = None,
    rows: int = 1024,
    cols: int = 512,
    interlaced: bool = False,
    diskbuffered: bool = False,
    extract_scan_repeats: bool = False,
    scalex: float = 0.01,
    slice_thickness: float = 0.05,
    apply_registration: bool = False,
    *,
    meta: DicomMetadata | None = None,
    overrides: t.Mapping[str, t.Any] | None = None,
) -> list:
    """Creates a DICOM file with the data parsed from
    the input file.

    Args:
            input_file: File with OCT data, .fda/.fds/.img/.e2e/.OCT
            output_dir: Output directory, will be created if
            not currently exists. Default None places file in
            current working directory.
            rows: If .img file, allows for manually setting rows
            cols: If .img file, allows for manually setting cols
            interlaced: If .img file, allows for setting interlaced
            diskbuffered: If Bioptigen .OCT, allows for setting diskbuffered
            extract_scan_repeats: If .e2e file, allows for extracting all scan repeats
            scalex: If .e2e file, allows for manually setting x scale (in mm)
            slice_thickness: If .e2e file, allows for manually setting z scale (in mm)
            apply_registration: If .e2e file, apply Heidelberg B-scan
                registration to OCT pixels and layer contours before writing
                DICOM. Defaults to False.
            meta: ``DicomMetadata`` to write instead of the metadata built
                from this file. The object is copied and is not modified.
            overrides: DICOM keywords applied to a copy of ``meta`` (or of
                the metadata built from the file). ``StudyInstanceUID`` is
                one UID for every file this call writes. ``SeriesInstanceUID``
                is a UID for the structural OPT, or a dict of roles
                ``opt``, ``fundus``, ``fundus_grayscale``, ``octa``, and
                ``seg`` to a UID or a list of UIDs. See ``prepare_export``.

    Returns:
            list: list of Path(s) to DICOM file
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()

    study = read_study(
        input_file,
        rows=rows,
        cols=cols,
        interlaced=interlaced,
        diskbuffered=diskbuffered,
        extract_scan_repeats=extract_scan_repeats,
        scalex=scalex,
        slice_thickness=slice_thickness,
        apply_registration=apply_registration,
    )
    return write_dicom_study(study, output_dir, meta=meta, overrides=overrides)


def create_dicom_from_boct(
    input_file: str,
    output_dir: str = None,
    diskbuffered: bool = False,
    *,
    meta: DicomMetadata | None = None,
    overrides: t.Mapping[str, t.Any] | None = None,
) -> list:
    """Read a Bioptigen OCT file into a ``DicomStudy`` and write it."""
    study = read_boct_study(input_file, diskbuffered=diskbuffered)
    return write_dicom_study(study, output_dir, meta=meta, overrides=overrides)


def create_dicom_from_e2e(
    input_file: str,
    output_dir: str = None,
    extract_scan_repeats: bool = False,
    scalex: float = 0.01,
    slice_thickness: float = 0.05,
    apply_registration: bool = False,
    *,
    meta: DicomMetadata | None = None,
    overrides: t.Mapping[str, t.Any] | None = None,
) -> list:
    """Read a Heidelberg E2E file into a ``DicomStudy`` and write it."""
    study = read_e2e_study(
        input_file,
        extract_scan_repeats=extract_scan_repeats,
        scalex=scalex,
        slice_thickness=slice_thickness,
        apply_registration=apply_registration,
    )
    return write_dicom_study(study, output_dir, meta=meta, overrides=overrides)


def create_dicom_from_fda(
    input_file: str,
    output_dir: str,
    *,
    meta: DicomMetadata | None = None,
    overrides: t.Mapping[str, t.Any] | None = None,
) -> list:
    """Read a Topcon FDA file into a ``DicomStudy`` and write it.

    Contours become a height-map segmentation. An angiography chunk becomes
    a B-scan Volume Analysis companion. JPEG 2000 localizers are encapsulated;
    a colour fundus stored as B, G, R is written as RGB.

    Args:
            input_file: FDA file with OCT data
            output_dir: Output directory
            meta: ``DicomMetadata`` used for every instance instead of the
                metadata parsed from this file. It is copied and not modified.
            overrides: DICOM keywords applied at write time. See
                ``create_dicom_from_oct``.

    Returns:
            list: List of path(s) to DICOM file(s)
    """
    study = read_fda_study(input_file)
    return write_dicom_study(study, output_dir, meta=meta, overrides=overrides)


def create_dicom_from_fds(
    input_file: str,
    output_dir: str,
    *,
    meta: DicomMetadata | None = None,
    overrides: t.Mapping[str, t.Any] | None = None,
) -> list:
    """Read a Topcon FDS file into a ``DicomStudy`` and write it."""
    study = read_fds_study(input_file)
    return write_dicom_study(study, output_dir, meta=meta, overrides=overrides)


def create_dicom_from_img(
    input_file: str,
    output_dir: str,
    rows: int = 1024,
    cols: int = 512,
    interlaced: bool = False,
    *,
    meta: DicomMetadata | None = None,
    overrides: t.Mapping[str, t.Any] | None = None,
) -> list:
    """Read a Zeiss IMG file into a ``DicomStudy`` and write it."""
    study = read_img_study(input_file, rows=rows, cols=cols, interlaced=interlaced)
    return write_dicom_study(study, output_dir, meta=meta, overrides=overrides)


def create_dicom_from_poct(
    input_file: str,
    output_dir: str,
    *,
    meta: DicomMetadata | None = None,
    overrides: t.Mapping[str, t.Any] | None = None,
) -> list:
    """Read an Optovue OCT file into a ``DicomStudy`` and write it."""
    study = read_poct_study(input_file)
    return write_dicom_study(study, output_dir, meta=meta, overrides=overrides)
