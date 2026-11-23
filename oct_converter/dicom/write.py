"""Write a ``DicomStudy`` to DICOM files.

The study is already format-neutral. This module only encodes it.
"""

from __future__ import annotations

import copy
import typing as t
from collections import defaultdict
from pathlib import Path

from oct_converter.dicom.metadata import DicomMetadata
from oct_converter.dicom.study import DicomStudy, ExportPhoto


def write_dicom_study(
    study: DicomStudy,
    output_dir: str | Path | None = None,
    *,
    meta: DicomMetadata | None = None,
    overrides: t.Mapping[str, t.Any] | None = None,
) -> list[Path]:
    """Write every photo, OPT, OCTA volume, and segmentation in ``study``.

    ``meta``, when given, replaces each object's parsed metadata. ``overrides``
    are applied to a copy. The objects on ``study`` are not modified.
    One Study Instance UID is shared by every file. ``localizer_geometry`` is
    applied after that copy so a replacement ``meta`` still carries the scan.
    """
    from oct_converter.dicom.dicom import (  # lazy: dicom imports this module
        _begin_export,
        _with_acquisition_device,
        _with_description,
        _without_opt_image_plane,
        write_color_fundus_dicom,
        write_fundus_dicom,
        write_heightmap_seg_dicom,
        write_oct_bscan_volume_analysis_dicom,
        write_opt_dicom,
    )
    from oct_converter.dicom.metadata import OPTAcquisitionDevice

    output_dir = Path.cwd() if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    supplied = meta
    shared_study: str | None = None
    plan = None
    role_index: dict[str, int] = defaultdict(int)

    def export_meta(built: DicomMetadata) -> DicomMetadata:
        nonlocal shared_study, plan
        working, shared_study, plan = _begin_export(
            built, supplied, overrides, study_instance_uid=shared_study
        )
        return working

    def series_uid(role: str) -> str | None:
        index = role_index[role]
        role_index[role] += 1
        if plan is None:
            return None
        return plan.for_role(role, index)

    def write_photo(working: DicomMetadata, photo: ExportPhoto):
        path = output_dir / photo.filename
        uid = series_uid(photo.role)

        def emit():
            writer = write_color_fundus_dicom if photo.color else write_fundus_dicom
            if photo.jpeg2000 is not None:
                return writer(
                    working,
                    None,
                    path,
                    study_instance_uid=shared_study,
                    series_instance_uid=uid,
                    compressed_frames=[photo.jpeg2000],
                )
            return writer(
                working,
                photo.image,
                path,
                study_instance_uid=shared_study,
                series_instance_uid=uid,
            )

        call = emit
        if photo.strip_opt_geometry:
            inner = call
            call = lambda inner=inner: _without_opt_image_plane(working, inner)
        if photo.fundus_camera:
            inner = call
            call = lambda inner=inner: _with_acquisition_device(
                working, OPTAcquisitionDevice.FundusCamera, inner
            )
        if photo.description_fallback is not None:
            inner = call
            call = lambda inner=inner: _with_description(
                working, photo.description_fallback, inner
            )
        return call(), path

    photo_paths: list[Path] = []
    fundus_by_id: dict = {}
    for photo in study.photos:
        if photo.meta is None:
            raise ValueError(f"{photo.filename} has no metadata")
        dataset, path = write_photo(export_meta(photo.meta), photo)
        photo_paths.append(path)
        if photo.image_id:
            fundus_by_id[photo.image_id] = dataset

    series_paths: list[Path] = []
    for oct_series in study.series:
        working = export_meta(oct_series.meta)
        if oct_series.localizer_geometry is not None:
            working.scan_geometry = copy.deepcopy(oct_series.localizer_geometry)

        localizer_ds = None
        localizer_path = None
        if oct_series.localizer is not None:
            localizer_ds, localizer_path = write_photo(working, oct_series.localizer)
        elif oct_series.linked_photo_id:
            localizer_ds = fundus_by_id.get(oct_series.linked_photo_id)

        opt_path = output_dir / oct_series.filename
        frames = None if oct_series.compressed_frames else oct_series.volume

        def write_opt(
            working=working,
            frames=frames,
            localizer_ds=localizer_ds,
            opt_path=opt_path,
            compressed=oct_series.compressed_frames,
        ):
            return write_opt_dicom(
                working,
                frames,
                opt_path,
                study_instance_uid=shared_study,
                series_instance_uid=series_uid("opt"),
                fundus_ds=localizer_ds,
                compressed_frames=compressed,
            )

        if oct_series.description_fallback is not None:
            opt_ds = _with_description(working, oct_series.description_fallback, write_opt)
        else:
            opt_ds = write_opt()
        series_paths.append(opt_path)

        if oct_series.angiography:
            angio_path = output_dir / oct_series.angiography_filename

            def write_angio(
                working=working,
                angio_path=angio_path,
                opt_ds=opt_ds,
            ):
                return _with_description(
                    working,
                    oct_series.angiography_description or "",
                    lambda: write_oct_bscan_volume_analysis_dicom(
                        working,
                        oct_series.angiography,
                        opt_ds,
                        angio_path,
                        bscans_per_frame=oct_series.bscans_per_frame,
                        series_instance_uid=series_uid("octa"),
                    ),
                )

            write_angio()
            series_paths.append(angio_path)

        if oct_series.contours and oct_series.segmentation_filename:
            seg_path = write_heightmap_seg_dicom(
                working,
                oct_series.contours,
                opt_ds,
                output_dir / oct_series.segmentation_filename,
                axial_spacing_mm=oct_series.axial_spacing_mm,
                series_instance_uid=series_uid("seg"),
            )
            if seg_path is not None:
                series_paths.append(Path(seg_path))

        if localizer_path is not None:
            series_paths.append(localizer_path)
        for photo in oct_series.photos:
            _, path = write_photo(working, photo)
            series_paths.append(path)

    return photo_paths + series_paths
