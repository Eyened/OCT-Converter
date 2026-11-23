"""Topcon scan geometry and colour-fundus byte order.

Used while parsing a file into a ``DicomStudy``. The DICOM writer reads the
resulting geometry and pixels and does not inspect Topcon chunks.
"""

from __future__ import annotations

import math

import numpy as np

from oct_converter.dicom.metadata import ScanGeometryMeta


def localizer_box(metadata: dict | None, key: str) -> list[float] | None:
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


def scan_field(oct) -> tuple[int | None, float | None, float | None]:
    """IMG-header scan mode and the PARAM_SCAN field size, in millimetres."""
    header = oct.oct_header or {}
    return header.get("scan_mode"), header.get("x_dimension_mm"), header.get("y_dimension_mm")


def localizer_geometry(
    box: list[float],
    n_frames: int,
    scan_mode: int | None = 2,
    x_mm: float | None = None,
    y_mm: float | None = None,
) -> ScanGeometryMeta:
    """Fundus-relative B-scan lines for one Topcon scan mode."""
    mode = int(scan_mode) if scan_mode is not None else 2
    if mode == 0:
        lines = _line(box)
        scan_type = "linear"
    elif mode == 1:
        return ScanGeometryMeta(
            scan_type="circular",
            centre=[box[0], box[1]],
            radius=float(box[2]),
            start_angle=0.0,
        )
    elif mode == 3:
        lines = _radial_lines(box, n_frames)
        scan_type = "linear"
    elif mode == 4:
        lines = _two_line_cross(box)
        scan_type = "linear"
    elif mode == 6:
        lines = _vertical_bscan_lines(box, n_frames)
        scan_type = "volume"
    elif mode == 11:
        lines = _five_line_cross(box, float(x_mm or 0), float(y_mm or 0))
        scan_type = "linear"
    else:
        lines = _horizontal_bscan_lines(box, n_frames)
        scan_type = "volume"
    return ScanGeometryMeta(
        scan_type=scan_type,
        line_start=list(lines[0]["line_start"]),
        line_end=list(lines[0]["line_end"]),
        frame_lines=lines,
    )


def bgr_display(payload: bytes) -> np.ndarray | None:
    """OpenCV BGR view of a colour fundus whose components are stored B, G, R.

    Returns None when the image cannot be decoded or when the red and blue
    components match, which is how an infrared localizer is stored.
    """
    import cv2

    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        return None
    if np.array_equal(decoded[..., 0], decoded[..., 2]):
        return None
    return np.ascontiguousarray(decoded)


def _horizontal_bscan_lines(box: list[float], n_frames: int) -> list[dict]:
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


def _vertical_bscan_lines(box: list[float], n_frames: int) -> list[dict]:
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


def _line(box: list[float]) -> list[dict]:
    """A line scan stores the B-scan endpoints in the registration box."""
    x1, y1, x2, y2 = box
    return [{"line_start": [x1, y1], "line_end": [x2, y2]}]


def _two_line_cross(box: list[float]) -> list[dict]:
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


def _five_line_cross(box: list[float], x_mm: float, y_mm: float) -> list[dict]:
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


def _radial_lines(box: list[float], n_frames: int) -> list[dict]:
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
