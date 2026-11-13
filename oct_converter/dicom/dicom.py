from __future__ import annotations

import math
import typing as t
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
from construct import StreamError, StringError
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import (
    UID,
    ExplicitVRLittleEndian,
    JPEG2000,
    JPEG2000Lossless,
    OphthalmicPhotography8BitImageStorage,
    OphthalmicPhotography16BitImageStorage,
    OphthalmicTomographyImageStorage,
    generate_uid,
)
from pydicom.valuerep import DSfloat

from oct_converter.dicom.boct_meta import boct_dicom_metadata
from oct_converter.dicom.e2e_meta import e2e_dicom_metadata
from oct_converter.dicom.fda_meta import fda_dicom_metadata, opt_series_fallback
from oct_converter.dicom.j2k import J2KFrames, is_jpeg2000, prepare_j2k_frames
from oct_converter.dicom.fds_meta import fds_dicom_metadata
from oct_converter.dicom.heightmap import (
    HEIGHTMAP_PADDING_VALUE,
    contours_to_heightmaps,
)
from oct_converter.dicom.img_meta import img_dicom_metadata
from oct_converter.dicom.metadata import (
    DicomMetadata,
    OCTDetectorType,
    OPTAcquisitionDevice,
    ScanGeometryMeta,
)
from oct_converter.dicom.poct_meta import poct_dicom_metadata
from oct_converter.exceptions import InvalidOCTReaderError
from oct_converter.readers import BOCT, E2E, FDA, FDS, IMG, POCT
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
        pixel_data, bits = _encode_pixel_array(frames)
        if pixel_data.ndim == 2:
            pixel_data = pixel_data[np.newaxis, ...]
        if pixel_data.ndim != 3:
            raise ValueError(
                f"Expected OCT frames as a 3-D volume, got shape {pixel_data.shape}"
            )
        pixel_representation = (
            1 if np.issubdtype(pixel_data.dtype, np.signedinteger) else 0
        )
        _apply_pixel_encoding_tags(
            ds,
            bits,
            pixel_representation=pixel_representation,
        )
        n_frames = int(pixel_data.shape[0])
        rows = int(pixel_data.shape[1])
        columns = int(pixel_data.shape[2])
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
    if j2k is None:
        ds.PixelData = _pixel_data_bytes(pixel_data)
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
    ds.SeriesInstanceUID = generate_uid()
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
        pixel_data, bits = _encode_pixel_array(_as_grayscale_array(frames))
        if pixel_data.ndim != 2:
            raise ValueError(
                f"Expected 2-D grayscale fundus image, got shape {pixel_data.shape}"
            )
        rows = int(pixel_data.shape[0])
        columns = int(pixel_data.shape[1])
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
    if j2k is not None:
        _apply_j2k_pixels(ds, j2k)
    else:
        pixel_representation = (
            1 if np.issubdtype(pixel_data.dtype, np.signedinteger) else 0
        )
        _apply_pixel_encoding_tags(
            ds,
            bits,
            pixel_representation=pixel_representation,
        )
    ds.NumberOfFrames = 1

    # Multi-frame Functional Groups Module PS3.3 C.7.6.16
    _photography_timing(ds, meta)
    ds.InstanceNumber = 1
    ds.Rows = rows
    ds.Columns = columns

    if j2k is None:
        ds.PixelData = _pixel_data_bytes(pixel_data)
    ds.save_as(
        filepath, implicit_vr=False, little_endian=True, enforce_file_format=True
    )
    return ds


def write_color_fundus_dicom(
    meta: DicomMetadata,
    frames: t.Sequence[np.ndarray] | None,
    filepath: Path,
    study_instance_uid: str | None = None,
    compressed_frames: t.Sequence[bytes] | None = None,
) -> Path:
    """Writes required DICOM metadata and RGB fundus pixel data to .dcm file.

    Args:
            meta: DICOM metadata information
            frames: list of frames of pixel data
            filepath: Path to where output file is being saved
            study_instance_uid: Optional shared Study Instance UID
            compressed_frames: JPEG 2000 payloads to encapsulate. ``frames``
                is ignored when this is set. Three-component images only.
    Returns:
            FileDataset of the written fundus instance (path is ``filepath``)
    """
    ds = opt_base_dicom(filepath)
    ds = populate_patient_info(ds, meta)
    ds = populate_manufacturer_info(ds, meta)
    ds = populate_opt_series(ds, meta, study_instance_uid=study_instance_uid)
    ds.Modality = "OP"
    j2k = prepare_j2k_frames(list(compressed_frames)) if compressed_frames else None
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
        rgb = np.asarray(frames)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        pixel_data, bits = _encode_pixel_array(rgb)
        if pixel_data.ndim != 3 or pixel_data.shape[-1] != 3:
            raise ValueError(
                f"Expected HxWx3 RGB fundus image, got shape {pixel_data.shape}"
            )
        rows = int(pixel_data.shape[0])
        columns = int(pixel_data.shape[1])
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
    if j2k is not None:
        _apply_j2k_pixels(ds, j2k)
    else:
        pixel_representation = (
            1 if np.issubdtype(pixel_data.dtype, np.signedinteger) else 0
        )
        _apply_pixel_encoding_tags(
            ds,
            bits,
            samples_per_pixel=3,
            photometric="RGB",
            pixel_representation=pixel_representation,
        )
    ds.NumberOfFrames = 1

    _photography_timing(ds, meta)
    ds.InstanceNumber = 1
    ds.Rows = rows
    ds.Columns = columns

    if j2k is None:
        ds.PixelData = _pixel_data_bytes(pixel_data)
    ds.save_as(
        filepath, implicit_vr=False, little_endian=True, enforce_file_format=True
    )
    return ds


def _topcon_localizer_box(metadata: dict | None, key: str) -> list[float] | None:
    """Scan rectangle on a localizer, as [left, top, right, bottom] pixels.

    Topcon's own OPT uses the ``@REGIST_INFO`` box. ``@EFFECTIVE_SCAN_RANGE``
    is the fallback when registration is absent.
    """
    if not metadata:
        return None
    regist = metadata.get("regist_info") or {}
    regist_key = {
        "bounding_box_fundus_pixel": "bounding_box_in_fundus_pixels",
        "bounding_box_trc_pixel": "bounding_box_in_trc_pixels",
    }.get(key)
    box = regist.get(regist_key) if regist_key else None
    if not box or len(box) != 4:
        box = (metadata.get("effective_scan_range") or {}).get(key)
    if not box or len(box) != 4:
        return None
    return [float(v) for v in box]


def _set_image_plane(ds: Dataset, meta: DicomMetadata) -> None:
    """Write pixel spacing and orientation only when the source measured them."""
    spacing = meta.image_geometry.pixel_spacing
    if spacing:
        ds.PixelSpacing = spacing
    orientation = meta.image_geometry.image_orientation
    if orientation:
        ds.ImageOrientationPatient = orientation


def _topcon_horizontal_bscan_lines(box: list[float], n_frames: int) -> list[dict]:
    """Horizontal B-scan lines, matching Triton OPT localizer coordinates.

    Triton cube exports keep the column pair fixed and interpolate the row
    from the top of the box to the bottom, rounded to the nearest pixel.
    ``line_start`` / ``line_end`` are (column, row).
    """
    left, top, right, bottom = box
    lines = []
    for i in range(n_frames):
        if n_frames <= 1:
            row = round((top + bottom) / 2)
        else:
            row = round(top + (bottom - top) * i / (n_frames - 1))
        lines.append({"line_start": [left, row], "line_end": [right, row]})
    return lines


def _topcon_vertical_bscan_lines(box: list[float], n_frames: int) -> list[dict]:
    """Vertical B-scans stepping from the left of the box to the right."""
    left, top, right, bottom = box
    lines = []
    for i in range(n_frames):
        if n_frames <= 1:
            col = round((left + right) / 2)
        else:
            col = round(left + (right - left) * i / (n_frames - 1))
        lines.append({"line_start": [col, top], "line_end": [col, bottom]})
    return lines


def _topcon_line(box: list[float]) -> list[dict]:
    """A line scan stores the B-scan endpoints in the registration box."""
    x1, y1, x2, y2 = box
    return [{"line_start": [x1, y1], "line_end": [x2, y2]}]


def _topcon_two_line_cross(box: list[float]) -> list[dict]:
    """Horizontal then vertical line through the centre of a two-line cross.

    Mode 4 stores ``(centre column, centre row, half-length in pixels, 0)``.
    A rectangular box is the fallback.
    """
    a, b, c, d = box
    if d == 0:
        return [
            {"line_start": [a - c, b], "line_end": [a + c, b]},
            {"line_start": [a, b - c], "line_end": [a, b + c]},
        ]
    left, top, right, bottom = box
    mid_x = (left + right) / 2
    mid_y = (top + bottom) / 2
    return [
        {"line_start": [left, mid_y], "line_end": [right, mid_y]},
        {"line_start": [mid_x, top], "line_end": [mid_x, bottom]},
    ]


def _topcon_five_line_cross(
    box: list[float], x_mm: float, y_mm: float
) -> list[dict]:
    """Five horizontal lines, then five vertical lines.

    The registration box is the centre horizontal line. ``y_mm`` is the gap
    between the parallel lines. Topcon's own OPT places that gap by
    truncating ``y_mm / x_mm * line length`` to an integer pixel step.
    """
    x1, y1, x2, y2 = box
    span = abs(x2 - x1)
    step = int(y_mm / x_mm * span) if x_mm else 0
    lines = [
        {"line_start": [x1, y1 + k * step], "line_end": [x2, y2 + k * step]}
        for k in range(-2, 3)
    ]
    mid_x = int(round((x1 + x2) / 2))
    mid_y = int(round((y1 + y2) / 2))
    half = int(abs(round(x2 - x1))) // 2
    lines.extend(
        {
            "line_start": [mid_x + k * step, mid_y - half],
            "line_end": [mid_x + k * step, mid_y + half],
        }
        for k in range(-2, 3)
    )
    return lines


def _topcon_radial_lines(box: list[float], n_frames: int) -> list[dict]:
    """Equal-angle lines through the scan centre."""
    a, b, c, d = box
    if d == 0:
        cx, cy, half = a, b, float(c)
    else:
        left, top, right, bottom = box
        cx = (left + right) / 2
        cy = (top + bottom) / 2
        half = min(abs(right - left), abs(bottom - top)) / 2
    lines = []
    count = max(n_frames, 1)
    for i in range(count):
        theta = math.pi * i / count
        dx = half * math.cos(theta)
        dy = half * math.sin(theta)
        lines.append(
            {"line_start": [cx - dx, cy - dy], "line_end": [cx + dx, cy + dy]}
        )
    return lines


def _topcon_scan_field(oct) -> tuple[int | None, float | None, float | None]:
    """IMG-header scan mode and the PARAM_SCAN field size, in millimetres."""
    header = oct.oct_header or {}
    return header.get("scan_mode"), header.get("x_dimension_mm"), header.get("y_dimension_mm")


def _set_topcon_localizer(
    meta: DicomMetadata,
    box: list[float],
    n_frames: int,
    scan_mode: int | None = 2,
    x_mm: float | None = None,
    y_mm: float | None = None,
) -> None:
    mode = int(scan_mode) if scan_mode is not None else 2
    if mode == 0:
        lines = _topcon_line(box)
        scan_type = "linear"
    elif mode == 1:
        meta.scan_geometry = ScanGeometryMeta(
            scan_type="circular",
            centre=[box[0], box[1]],
            radius=float(box[2]),
            start_angle=0.0,
        )
        return
    elif mode == 3:
        lines = _topcon_radial_lines(box, n_frames)
        scan_type = "linear"
    elif mode == 4:
        lines = _topcon_two_line_cross(box)
        scan_type = "linear"
    elif mode == 6:
        lines = _topcon_vertical_bscan_lines(box, n_frames)
        scan_type = "volume"
    elif mode == 11:
        lines = _topcon_five_line_cross(box, float(x_mm or 0), float(y_mm or 0))
        scan_type = "linear"
    else:
        lines = _topcon_horizontal_bscan_lines(box, n_frames)
        scan_type = "volume"
    meta.scan_geometry = ScanGeometryMeta(
        scan_type=scan_type,
        line_start=list(lines[0]["line_start"]),
        line_end=list(lines[0]["line_end"]),
        frame_lines=lines,
    )


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
    """Use the FDA capture label, or a name for this object when it is blank."""
    saved = meta.series_info.description
    label = (meta.series_info.protocol or saved or "").strip()
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

    Returns:
            list: list of Path(s) to DICOM file
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()

    file_suffix = input_file.split(".")[-1].lower()

    if file_suffix == "fds":
        files = create_dicom_from_fds(input_file, output_dir)
    elif file_suffix == "fda":
        files = create_dicom_from_fda(input_file, output_dir)
    elif file_suffix == "img":
        files = create_dicom_from_img(input_file, output_dir, rows, cols, interlaced)
    elif file_suffix == "oct":
        # Bioptigen and Octovue both use .OCT.
        # BOCT._validate on init can check if Bioptigen, else Optivue
        try:
            BOCT(input_file)
            files = create_dicom_from_boct(input_file, output_dir, diskbuffered)
        except (InvalidOCTReaderError, StreamError, UnicodeDecodeError, StringError):
            # if BOCT raises, treat as POCT
            files = create_dicom_from_poct(input_file, output_dir)
    elif file_suffix == "e2e":
        files = create_dicom_from_e2e(
            input_file,
            output_dir,
            extract_scan_repeats,
            scalex,
            slice_thickness,
            apply_registration=apply_registration,
        )
    else:
        raise TypeError(
            f"DICOM conversion for {file_suffix} is not supported. "
            "Currently supported filetypes are .e2e, .fds, .fda, .img, .OCT."
        )

    return files


def create_dicom_from_boct(
    input_file: str,
    output_dir: str = None,
    diskbuffered: bool = False,
) -> list:
    """Creates DICOM file(s) with the data parsed from
    the input file.

    Args:
            input_file: Bioptigen OCT file
            output_dir: Output directory
            diskbuffered: If True, reduces memory usage by storing volume on disk using HDF5.

    Returns:
            list: List of path(s) to DICOM file(s)"""

    boct = BOCT(input_file)
    oct_volumes = boct.read_oct_volume(diskbuffered)
    if len(oct_volumes) == 0:
        raise ValueError("No OCT volumes found in OCT input file.")

    files = []

    for count, oct in enumerate(oct_volumes):
        meta = boct_dicom_metadata(oct)
        filename = f"{Path(input_file).stem}_{str(count)}.dcm"
        filepath = Path(output_dir, filename)
        file = write_opt_dicom(meta, oct.volume, filepath)
        files.append(filepath)

    return files


def create_dicom_from_e2e(
    input_file: str,
    output_dir: str = None,
    extract_scan_repeats: bool = False,
    scalex: float = 0.01,
    slice_thickness: float = 0.05,
    apply_registration: bool = False,
) -> list:
    """Creates DICOM file(s) with the data parsed from
    the input file.

    Args:
            input_file: E2E file with OCT data
            output_dir: Output directory
            extract_scan_repeats: If True, will extract all scan repeats
            scalex: Manually set scale of x axis
            slice_thickness: Manually set scale of z axis
            apply_registration: Apply Heidelberg B-scan registration to
                OCT pixels and contours before writing. Defaults to False.

    Returns:
            list: List of path(s) to DICOM file(s)
    """
    e2e = E2E(input_file)
    oct_volumes = e2e.read_oct_volume(
        scalex=scalex,
        slice_thickness=slice_thickness,
        apply_registration=apply_registration,
    )
    fundus_images = e2e.read_fundus_image(
        extract_scan_repeats=extract_scan_repeats, scalex=scalex
    )
    if len(oct_volumes) == 0 and len(fundus_images) == 0:
        raise ValueError("No OCT volumes or fundus images found in e2e input file.")

    files = []
    # Map series key -> fundus FileDataset for Ophthalmic Frame Location
    fundus_by_id: dict[str, Dataset] = {}
    # Shared Study UID per E2E file
    study_uid = generate_uid()

    if len(fundus_images) > 0:
        for count, fundus in enumerate(fundus_images):
            meta = e2e_dicom_metadata(fundus)
            filename = f"{Path(input_file).stem}_fundus_{str(count)}.dcm"
            filepath = Path(output_dir, filename)
            fundus_ds = write_fundus_dicom(
                meta, fundus.image, filepath, study_instance_uid=study_uid
            )
            files.append(filepath)
            if fundus.image_id:
                # Strip trailing underscores used for scan-repeat disambiguation
                key = fundus.image_id.rstrip("_")
                fundus_by_id.setdefault(key, fundus_ds)
                fundus_by_id[fundus.image_id] = fundus_ds

    if len(oct_volumes) > 0:
        for count, oct in enumerate(oct_volumes):
            meta = e2e_dicom_metadata(oct)
            for_uid = generate_uid()
            filename = f"{Path(input_file).stem}_oct_{str(count)}.dcm"
            filepath = Path(output_dir, filename)
            fundus_ds = None
            if oct.volume_id:
                fundus_ds = fundus_by_id.get(oct.volume_id) or fundus_by_id.get(
                    oct.volume_id.rstrip("_")
                )
            # Fallback: single fundus for single OCT
            if fundus_ds is None and len(fundus_by_id) == 1:
                fundus_ds = next(iter(fundus_by_id.values()))
            opt_ds = write_opt_dicom(
                meta,
                oct.volume,
                filepath,
                study_instance_uid=study_uid,
                frame_of_reference_uid=for_uid,
                fundus_ds=fundus_ds,
            )
            files.append(filepath)

            if oct.contours:
                # Axial spacing is scaley (E2E pixel_spacing[1]).
                axial_mm = None
                if oct.pixel_spacing and len(oct.pixel_spacing) >= 2:
                    axial_mm = float(oct.pixel_spacing[1])
                seg_filename = f"{Path(input_file).stem}_seg_{str(count)}.dcm"
                seg_filepath = Path(output_dir, seg_filename)
                seg_path = write_heightmap_seg_dicom(
                    meta,
                    oct.contours,
                    opt_ds,
                    seg_filepath,
                    axial_spacing_mm=axial_mm,
                )
                if seg_path is not None:
                    files.append(seg_path)

    return files


def create_dicom_from_fda(
    input_file: str,
    output_dir: str,
) -> list:
    """Creates DICOM file(s) with the data parsed from
    the input file.

    When layer contours are present on the OCT volume, also writes a Height
    Map Segmentation companion (``{stem}_seg.dcm``) sharing Study and Frame
    of Reference UIDs with the OPT volume.

    ``@IMG_JPEG`` and the infrared ``@IMG_TRC_02`` codestreams are encapsulated
    without being decoded. A colour ``@IMG_FUNDUS`` whose components are B, G, R
    is written as uncompressed RGB so photometric interpretation RGB is true.
    ``@IMG_MOT_COMP_03`` volumes, which are already raw pixels, stay
    uncompressed.

    Args:
            input_file: FDA file with OCT data
            output_dir: Output directory

    Returns:
            list: List of path(s) to DICOM file(s)
    """
    files = []
    fda = FDA(input_file)
    oct = fda.read_oct_volume(decode=False)
    compressed_oct = _jpeg2000_volume(oct.volume)
    if compressed_oct is None and _is_byte_volume(oct.volume):
        # IMG_JPEG that is not JPEG 2000 still has to be decoded.
        oct = fda.read_oct_volume()
    meta = fda_dicom_metadata(oct)
    study_uid = generate_uid()
    for_uid = generate_uid()
    stem = Path(input_file).stem
    n_frames = len(compressed_oct if compressed_oct is not None else oct.volume)
    scan_mode, x_mm, y_mm = _topcon_scan_field(oct)

    # Write the localizer before the OPT so each B-scan can reference it.
    # Triton cube exports reference the colour fundus, not the IR localizer.
    fundus_bytes = fda.read_fundus_image_bytes()
    fundus_ds = None
    fundus_path = None
    box = _topcon_localizer_box(oct.metadata, "bounding_box_fundus_pixel")
    if fundus_bytes and box and is_jpeg2000(fundus_bytes):
        fundus_path = Path(output_dir, f"{stem}_fundus.dcm")
        fundus_ds = _write_jpeg2000_photo(
            meta, fundus_bytes, fundus_path, study_uid, "COLOR"
        )
        _set_topcon_localizer(meta, box, n_frames, scan_mode, x_mm, y_mm)
    elif box:
        fundus = fda.read_fundus_image()
        if fundus:
            fundus_path = Path(output_dir, f"{stem}_fundus.dcm")
            fundus_ds = _write_decoded_color_fundus(
                meta, fundus.image, fundus_path, study_uid
            )
            _set_topcon_localizer(meta, box, n_frames, scan_mode, x_mm, y_mm)

    opt_path = Path(output_dir, f"{stem}.dcm")
    opt_ds = _with_description(
        meta,
        opt_series_fallback(scan_mode, meta.series_info.octa),
        lambda: write_opt_dicom(
            meta,
            None if compressed_oct is not None else oct.volume,
            opt_path,
            study_instance_uid=study_uid,
            frame_of_reference_uid=for_uid,
            fundus_ds=fundus_ds,
            compressed_frames=compressed_oct,
        ),
    )
    files.append(opt_path)

    if oct.contours:
        axial_mm = None
        if oct.pixel_spacing and len(oct.pixel_spacing) >= 3:
            axial_mm = float(oct.pixel_spacing[2])
        seg_filepath = Path(output_dir, f"{stem}_seg.dcm")
        seg_path = write_heightmap_seg_dicom(
            meta,
            oct.contours,
            opt_ds,
            seg_filepath,
            axial_spacing_mm=axial_mm,
        )
        if seg_path is not None:
            files.append(seg_path)

    if fundus_path is not None:
        files.append(fundus_path)

    gray_bytes = fda.read_fundus_image_gray_scale_bytes()
    if gray_bytes and is_jpeg2000(gray_bytes):
        gray_path = Path(output_dir, f"{stem}_fundus_grayscale.dcm")
        _write_jpeg2000_photo(meta, gray_bytes, gray_path, study_uid, "IR")
        files.append(gray_path)
    else:
        fundus_grayscale = fda.read_fundus_image_gray_scale()
        if fundus_grayscale:
            gray_path = Path(output_dir, f"{stem}_fundus_grayscale.dcm")
            _with_description(
                meta,
                "IR",
                lambda: _without_opt_image_plane(
                    meta,
                    lambda: write_fundus_dicom(
                        meta,
                        fundus_grayscale.image,
                        gray_path,
                        study_instance_uid=study_uid,
                    ),
                ),
            )
            files.append(gray_path)

    return files


def _is_byte_volume(volume) -> bool:
    return bool(volume) and isinstance(volume[0], (bytes, bytearray))


def _jpeg2000_volume(volume) -> list[bytes] | None:
    """JPEG 2000 OCT payloads, or None when the volume is raw pixels."""
    if not _is_byte_volume(volume) or not is_jpeg2000(volume[0]):
        return None
    return list(volume)


def _topcon_bgr_display(payload: bytes) -> np.ndarray | None:
    """OpenCV BGR view of a Topcon colour fundus, or None when R and B match.

    Topcon JPEG 2000 colour components are stored B, G, R while the JP2 colour
    box says sRGB. OpenCV's BGR buffer is the byte order photometric RGB must
    store. Infrared localizers have identical components, so their codestream
    can be encapsulated as-is.
    """
    import cv2

    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        return None
    if np.array_equal(decoded[..., 0], decoded[..., 2]):
        return None
    return np.ascontiguousarray(decoded)


def _write_decoded_color_fundus(meta, image, path: Path, study_uid: str):
    """Write a decoded colour fundus as RGB from the camera, not the OCT scanner."""
    return _with_description(
        meta,
        "COLOR",
        lambda: _with_acquisition_device(
            meta,
            OPTAcquisitionDevice.FundusCamera,
            lambda: _without_opt_image_plane(
                meta,
                lambda: write_color_fundus_dicom(
                    meta, image, path, study_instance_uid=study_uid
                ),
            ),
        ),
    )


def _write_jpeg2000_photo(
    meta, payload: bytes, path: Path, study_uid: str, description: str
):
    """Encapsulate a JPEG 2000 localizer, unless its components are B, G, R."""
    samples = prepare_j2k_frames([payload]).samples

    def write():
        if samples == 3:
            display = _topcon_bgr_display(payload)
            if display is not None:
                return _with_acquisition_device(
                    meta,
                    OPTAcquisitionDevice.FundusCamera,
                    lambda: write_color_fundus_dicom(
                        meta, display, path, study_instance_uid=study_uid
                    ),
                )
        writer = write_fundus_dicom if samples == 1 else write_color_fundus_dicom
        return writer(
            meta,
            None,
            path,
            study_instance_uid=study_uid,
            compressed_frames=[payload],
        )

    return _with_description(
        meta,
        description,
        lambda: _without_opt_image_plane(meta, write),
    )


def create_dicom_from_fds(
    input_file: str,
    output_dir: str,
) -> list:
    """Creates DICOM file(s) with the data parsed from
    the input file.

    Args:
            input_file: FDS file with OCT data
            output_dir: Output directory

    Returns:
            list: List of path(s) to DICOM file(s)
    """
    files = []
    fds = FDS(input_file)
    oct = fds.read_oct_volume()
    meta = fds_dicom_metadata(oct)
    stem = Path(input_file).stem
    n_frames = len(oct.volume)
    scan_mode, x_mm, y_mm = _topcon_scan_field(oct)

    fundus = fds.read_fundus_image()
    fundus_ds = None
    fundus_path = None
    box = _topcon_localizer_box(oct.metadata, "bounding_box_fundus_pixel")
    if fundus and box:
        fundus_path = Path(output_dir, f"{stem}_fundus.dcm")
        fundus_ds = _without_opt_image_plane(
            meta,
            lambda: write_color_fundus_dicom(meta, fundus.image, fundus_path),
        )
        _set_topcon_localizer(meta, box, n_frames, scan_mode, x_mm, y_mm)

    opt_path = Path(output_dir, f"{stem}.dcm")
    _with_description(
        meta,
        opt_series_fallback(scan_mode, meta.series_info.octa),
        lambda: write_opt_dicom(meta, oct.volume, opt_path, fundus_ds=fundus_ds),
    )
    files.append(opt_path)
    if fundus_path is not None:
        files.append(fundus_path)

    return files


def create_dicom_from_img(
    input_file: str,
    output_dir: str,
    rows: int = 1024,
    cols: int = 512,
    interlaced: bool = False,
) -> Path:
    """Creates a DICOM file with the data parsed from
    the input file.

    Args:
            input_file: .img file with OCT data
            output_dir: Output directory
            rows: Optional, for manually setting rows. Default 1024.
            cols: Optional, for manually setting cols. Default 512.
            interlaced: Optional, for setting interlaced. Default False.

    Returns:
            list: List of path(s) to DICOM file(s)
    """
    img = IMG(input_file)
    oct = img.read_oct_volume(rows, cols, interlaced)
    meta = img_dicom_metadata(oct)
    output_filename = f"{Path(input_file).stem}.dcm"
    filepath = Path(output_dir, output_filename)
    file = write_opt_dicom(meta, oct.volume, filepath)
    return [filepath]


def create_dicom_from_poct(
    input_file: str,
    output_dir: str,
) -> list:
    """Creates DICOM file(s) with the data parsed from
    the input file.

    Args:
            input_file: File with POCT data
            output_dir: Output directory

    Returns:
            list: List of path(s) to DICOM file(s)
    """
    poct = POCT(input_file)
    octs = poct.read_oct_volume()
    files = []
    for count, oct in enumerate(octs):
        meta = poct_dicom_metadata(oct)
        filename = f"{Path(input_file).stem}_{str(count)}.dcm"
        filepath = Path(output_dir, filename)
        file = write_opt_dicom(meta, oct.volume, filepath)
        files.append(filepath)

    return files
