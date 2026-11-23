"""Shared DICOM study produced by a parser and consumed by the writer.

No file format and no DICOM encoding live here.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np

from oct_converter.dicom.metadata import DicomMetadata, ScanGeometryMeta


@dataclass
class ExportPhoto:
    """One ophthalmic photography instance.

    ``jpeg2000`` is a codestream to encapsulate. ``image`` is decoded pixels.
    ``color``, ``fundus_camera``, ``strip_opt_geometry``, and
    ``description_fallback`` say how to write either form. A parser resolves
    vendor byte order before the photo is stored here.
    """

    filename: str
    role: str = "fundus"
    meta: DicomMetadata | None = None
    image: np.ndarray | None = None
    jpeg2000: bytes | None = None
    color: bool = False
    strip_opt_geometry: bool = False
    fundus_camera: bool = False
    description_fallback: str | None = None
    image_id: str | None = None


@dataclass
class ExportOct:
    """One structural OPT volume and the companions derived from it."""

    meta: DicomMetadata
    filename: str
    volume: t.Sequence[np.ndarray] | None = None
    compressed_frames: list[bytes] | None = None
    description_fallback: str | None = None
    localizer: ExportPhoto | None = None
    photos: list[ExportPhoto] = field(default_factory=list)
    linked_photo_id: str | None = None
    localizer_geometry: ScanGeometryMeta | None = None
    angiography: t.Sequence[np.ndarray] | None = None
    angiography_filename: str | None = None
    angiography_description: str | None = None
    bscans_per_frame: int = 1
    contours: dict | None = None
    segmentation_filename: str | None = None
    axial_spacing_mm: float | None = None


@dataclass
class DicomStudy:
    """Photos and OPT series ready to write. No source-file details."""

    photos: list[ExportPhoto] = field(default_factory=list)
    series: list[ExportOct] = field(default_factory=list)
