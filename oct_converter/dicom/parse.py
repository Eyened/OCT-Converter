"""Build a ``DicomStudy`` from one proprietary OCT file.

Each function reads one format. None of them write DICOM, and none of them
import the DICOM writer.
"""

from __future__ import annotations

from pathlib import Path

from oct_converter.dicom.boct_meta import boct_dicom_metadata
from oct_converter.dicom.e2e_meta import e2e_dicom_metadata
from oct_converter.dicom.fda_meta import fda_dicom_metadata, opt_series_fallback
from oct_converter.dicom.fds_meta import fds_dicom_metadata
from oct_converter.dicom.img_meta import img_dicom_metadata
from oct_converter.dicom.j2k import is_byte_volume, is_jpeg2000, jpeg2000_volume, prepare_j2k_frames
from oct_converter.dicom.poct_meta import poct_dicom_metadata
from oct_converter.dicom.study import DicomStudy, ExportOct, ExportPhoto
from oct_converter.dicom.topcon import bgr_display, localizer_box, localizer_geometry, scan_field
from oct_converter.readers import BOCT, E2E, FDA, FDS, IMG, POCT


def _axial_mm(spacing, index: int) -> float | None:
    if spacing and len(spacing) > index:
        return float(spacing[index])
    return None


def _jpeg_photo(payload: bytes, filename: str, role: str, description: str) -> ExportPhoto:
    """A JPEG 2000 localizer, already decided as pixels or a codestream."""
    samples = prepare_j2k_frames([payload]).samples
    if samples == 3:
        display = bgr_display(payload)
        if display is not None:
            return ExportPhoto(
                filename=filename,
                role=role,
                image=display,
                color=True,
                strip_opt_geometry=True,
                fundus_camera=True,
                description_fallback=description,
            )
        return ExportPhoto(
            filename=filename,
            role=role,
            jpeg2000=payload,
            color=True,
            strip_opt_geometry=True,
            description_fallback=description,
        )
    return ExportPhoto(
        filename=filename,
        role=role,
        jpeg2000=payload,
        strip_opt_geometry=True,
        description_fallback=description,
    )


def _link_localizers(photos: list[ExportPhoto], series: list[ExportOct]) -> None:
    """Point each OCT at the fundus photo that belongs to it.

    Heidelberg ids sometimes differ by a trailing underscore. One photo in
    the file is the localizer for every volume that does not name its own.
    """
    by_id: dict[str, str] = {}
    for photo in photos:
        if not photo.image_id:
            continue
        by_id.setdefault(photo.image_id.rstrip("_"), photo.image_id)
        by_id[photo.image_id] = photo.image_id
    only = next(iter(by_id.values())) if len(by_id) == 1 else None
    for item in series:
        found = None
        if item.linked_photo_id:
            found = by_id.get(item.linked_photo_id) or by_id.get(
                item.linked_photo_id.rstrip("_")
            )
        item.linked_photo_id = found or only


def read_fda_study(input_file: str) -> DicomStudy:
    """FDA file to a structural series, optional localizer, OCTA, and segmentation."""
    fda = FDA(input_file)
    oct_volume = fda.read_oct_volume(decode=False)
    compressed = jpeg2000_volume(oct_volume.volume)
    if compressed is None and is_byte_volume(oct_volume.volume):
        oct_volume = fda.read_oct_volume()
    meta = fda_dicom_metadata(oct_volume)
    stem = Path(input_file).stem
    box = localizer_box(oct_volume.metadata, "bounding_box_fundus_pixel")
    mode, x_mm, y_mm = scan_field(oct_volume)
    n_frames = len(compressed if compressed is not None else oct_volume.volume)

    localizer = None
    fundus_bytes = fda.read_fundus_image_bytes()
    if fundus_bytes and box and is_jpeg2000(fundus_bytes):
        localizer = _jpeg_photo(fundus_bytes, f"{stem}_fundus.dcm", "fundus", "COLOR")
    elif box:
        fundus = fda.read_fundus_image()
        if fundus:
            localizer = ExportPhoto(
                filename=f"{stem}_fundus.dcm",
                image=fundus.image,
                color=True,
                strip_opt_geometry=True,
                fundus_camera=True,
                description_fallback="COLOR",
            )

    photos: list[ExportPhoto] = []
    gray_bytes = fda.read_fundus_image_gray_scale_bytes()
    if gray_bytes and is_jpeg2000(gray_bytes):
        photos.append(
            _jpeg_photo(gray_bytes, f"{stem}_fundus_grayscale.dcm", "fundus_grayscale", "IR")
        )
    else:
        gray = fda.read_fundus_image_gray_scale()
        if gray:
            photos.append(
                ExportPhoto(
                    filename=f"{stem}_fundus_grayscale.dcm",
                    role="fundus_grayscale",
                    image=gray.image,
                    strip_opt_geometry=True,
                    description_fallback="IR",
                )
            )

    angiography = fda.read_angiography_volume()
    bscans_per_frame = 1
    if angiography is not None:
        bscans_per_frame = fda.read_angiography_param_bscans_per_frame() or 1
    geometry = (
        localizer_geometry(box, n_frames, mode, x_mm, y_mm) if localizer and box else None
    )
    return DicomStudy(
        series=[
            ExportOct(
                meta=meta,
                filename=f"{stem}.dcm",
                volume=None if compressed is not None else oct_volume.volume,
                compressed_frames=compressed,
                description_fallback=opt_series_fallback(mode, meta.series_info.octa),
                localizer=localizer,
                photos=photos,
                localizer_geometry=geometry,
                angiography=angiography,
                angiography_filename=f"{stem}_octa.dcm",
                angiography_description="OCTA flow" if angiography is not None else None,
                bscans_per_frame=bscans_per_frame,
                contours=oct_volume.contours or None,
                segmentation_filename=f"{stem}_seg.dcm",
                axial_spacing_mm=_axial_mm(oct_volume.pixel_spacing, 2),
            )
        ]
    )


def read_fds_study(input_file: str) -> DicomStudy:
    """FDS file to a structural series plus an optional colour localizer."""
    fds = FDS(input_file)
    oct_volume = fds.read_oct_volume()
    meta = fds_dicom_metadata(oct_volume)
    stem = Path(input_file).stem
    box = localizer_box(oct_volume.metadata, "bounding_box_fundus_pixel")
    mode, x_mm, y_mm = scan_field(oct_volume)
    fundus = fds.read_fundus_image()
    localizer = None
    if fundus and box:
        localizer = ExportPhoto(
            filename=f"{stem}_fundus.dcm",
            image=fundus.image,
            color=True,
            strip_opt_geometry=True,
        )
    geometry = (
        localizer_geometry(box, len(oct_volume.volume), mode, x_mm, y_mm)
        if localizer and box
        else None
    )
    return DicomStudy(
        series=[
            ExportOct(
                meta=meta,
                filename=f"{stem}.dcm",
                volume=oct_volume.volume,
                description_fallback=opt_series_fallback(mode, meta.series_info.octa),
                localizer=localizer,
                localizer_geometry=geometry,
            )
        ]
    )


def read_e2e_study(
    input_file: str,
    *,
    extract_scan_repeats: bool = False,
    scalex: float = 0.01,
    slice_thickness: float = 0.05,
    apply_registration: bool = False,
) -> DicomStudy:
    """E2E file to study-level fundus photos and one OPT series per volume."""
    e2e = E2E(input_file)
    oct_volumes = e2e.read_oct_volume(
        scalex=scalex,
        slice_thickness=slice_thickness,
        apply_registration=apply_registration,
    )
    fundus_images = e2e.read_fundus_image(
        extract_scan_repeats=extract_scan_repeats, scalex=scalex
    )
    if not oct_volumes and not fundus_images:
        raise ValueError("No OCT volumes or fundus images found in e2e input file.")

    stem = Path(input_file).stem
    photos = [
        ExportPhoto(
            filename=f"{stem}_fundus_{count}.dcm",
            meta=e2e_dicom_metadata(fundus),
            image=fundus.image,
            image_id=fundus.image_id,
        )
        for count, fundus in enumerate(fundus_images)
    ]
    series = [
        ExportOct(
            meta=e2e_dicom_metadata(oct_volume),
            filename=f"{stem}_oct_{count}.dcm",
            volume=oct_volume.volume,
            linked_photo_id=oct_volume.volume_id,
            contours=oct_volume.contours or None,
            segmentation_filename=f"{stem}_seg_{count}.dcm",
            axial_spacing_mm=_axial_mm(oct_volume.pixel_spacing, 1),
        )
        for count, oct_volume in enumerate(oct_volumes)
    ]
    _link_localizers(photos, series)
    return DicomStudy(photos=photos, series=series)


def read_boct_study(input_file: str, *, diskbuffered: bool = False) -> DicomStudy:
    """Bioptigen OCT file to one OPT series per volume."""
    boct = BOCT(input_file)
    volumes = boct.read_oct_volume(diskbuffered)
    if not volumes:
        raise ValueError("No OCT volumes found in OCT input file.")
    stem = Path(input_file).stem
    return DicomStudy(
        series=[
            ExportOct(
                meta=boct_dicom_metadata(oct_volume),
                filename=f"{stem}_{count}.dcm",
                volume=oct_volume.volume,
            )
            for count, oct_volume in enumerate(volumes)
        ]
    )


def read_img_study(
    input_file: str,
    *,
    rows: int = 1024,
    cols: int = 512,
    interlaced: bool = False,
) -> DicomStudy:
    """Zeiss IMG file to one OPT series."""
    img = IMG(input_file)
    oct_volume = img.read_oct_volume(rows, cols, interlaced)
    return DicomStudy(
        series=[
            ExportOct(
                meta=img_dicom_metadata(oct_volume),
                filename=f"{Path(input_file).stem}.dcm",
                volume=oct_volume.volume,
            )
        ]
    )


def read_poct_study(input_file: str) -> DicomStudy:
    """Optovue OCT file to one OPT series per volume."""
    poct = POCT(input_file)
    volumes = poct.read_oct_volume()
    stem = Path(input_file).stem
    return DicomStudy(
        series=[
            ExportOct(
                meta=poct_dicom_metadata(oct_volume),
                filename=f"{stem}_{count}.dcm",
                volume=oct_volume.volume,
            )
            for count, oct_volume in enumerate(volumes)
        ]
    )


def read_study(
    input_file: str,
    *,
    rows: int = 1024,
    cols: int = 512,
    interlaced: bool = False,
    diskbuffered: bool = False,
    extract_scan_repeats: bool = False,
    scalex: float = 0.01,
    slice_thickness: float = 0.05,
    apply_registration: bool = False,
) -> DicomStudy:
    """Parse one supported OCT file into a ``DicomStudy``."""
    from construct import StreamError, StringError

    from oct_converter.exceptions import InvalidOCTReaderError

    suffix = input_file.split(".")[-1].lower()
    if suffix == "fda":
        return read_fda_study(input_file)
    if suffix == "fds":
        return read_fds_study(input_file)
    if suffix == "e2e":
        return read_e2e_study(
            input_file,
            extract_scan_repeats=extract_scan_repeats,
            scalex=scalex,
            slice_thickness=slice_thickness,
            apply_registration=apply_registration,
        )
    if suffix == "img":
        return read_img_study(input_file, rows=rows, cols=cols, interlaced=interlaced)
    if suffix == "oct":
        # Bioptigen and Optovue both use .OCT.
        try:
            BOCT(input_file)
        except (InvalidOCTReaderError, StreamError, UnicodeDecodeError, StringError):
            return read_poct_study(input_file)
        return read_boct_study(input_file, diskbuffered=diskbuffered)
    raise TypeError(
        f"DICOM conversion for {suffix} is not supported. "
        "Currently supported filetypes are .e2e, .fds, .fda, .img, .OCT."
    )
