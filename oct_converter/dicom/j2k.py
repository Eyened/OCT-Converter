"""JPEG 2000 payloads stored in Topcon .fda chunks.

FDA ``@IMG_JPEG``, ``@IMG_FUNDUS`` and ``@IMG_TRC_02`` chunks hold JPEG 2000
data, either as a raw codestream or as a JP2 file. DICOM encapsulation uses
the codestream only (no JP2 wrapper) and must advertise the wavelet the
codestream actually uses.
"""

from __future__ import annotations

from dataclasses import dataclass

_SOC = b"\xff\x4f"
_JP2_BRAND = b"jP  "
_SOT_SOD_EOC = frozenset({0x90, 0x93, 0xD9})


@dataclass(frozen=True)
class J2KFrames:
    """One or more JPEG 2000 frames that share a DICOM pixel description."""

    codestreams: tuple[bytes, ...]
    rows: int
    columns: int
    samples: int
    bits_allocated: int
    bits_stored: int
    pixel_representation: int
    reversible: bool
    mct: bool

    @property
    def photometric(self) -> str:
        if self.samples == 1:
            return "MONOCHROME2"
        if self.mct:
            return "YBR_RCT" if self.reversible else "YBR_ICT"
        return "RGB"


def is_jpeg2000(payload: bytes) -> bool:
    """True for a raw JPEG 2000 codestream or a JP2 file."""
    if len(payload) < 12:
        return False
    return payload.startswith(_SOC) or payload[4:8] == _JP2_BRAND


def prepare_j2k_frames(payloads: list[bytes]) -> J2KFrames:
    """Parse FDA image payloads into codestreams with one shared description.

    JP2 files contribute the inner codestream box. Raw codestreams are kept
    as they are. Frames are not decoded.
    """
    if not payloads:
        raise ValueError("No JPEG 2000 frames to encapsulate.")

    parsed: list[tuple[bytes, dict]] = []
    for index, payload in enumerate(payloads):
        codestream = _codestream(payload)
        parsed.append((codestream, _main_header(codestream, index)))

    first = parsed[0][1]
    for index, (_, info) in enumerate(parsed[1:], start=1):
        if info != first:
            raise ValueError(
                f"JPEG 2000 frame {index} does not match frame 0 "
                f"({info} != {first})."
            )

    return J2KFrames(
        codestreams=tuple(_even_length(codestream) for codestream, _ in parsed),
        rows=first["rows"],
        columns=first["columns"],
        samples=first["samples"],
        bits_allocated=first["bits_allocated"],
        bits_stored=first["bits_stored"],
        pixel_representation=first["pixel_representation"],
        reversible=first["reversible"],
        mct=first["mct"],
    )


def _codestream(payload: bytes) -> bytes:
    if payload.startswith(_SOC):
        return payload
    if len(payload) >= 8 and payload[4:8] == _JP2_BRAND:
        return _jp2_codestream(payload)
    raise ValueError("Image payload is not a JPEG 2000 codestream or JP2 file.")


def _jp2_codestream(data: bytes) -> bytes:
    """Return the ``jp2c`` box payload, which is the raw codestream."""
    offset = 0
    end = len(data)
    while offset + 8 <= end:
        length = int.from_bytes(data[offset : offset + 4], "big")
        box_type = data[offset + 4 : offset + 8]
        header = 8
        if length == 1:
            if offset + 16 > end:
                break
            length = int.from_bytes(data[offset + 8 : offset + 16], "big")
            header = 16
        elif length == 0:
            length = end - offset
        if length < header or offset + length > end:
            break
        if box_type == b"jp2c":
            codestream = data[offset + header : offset + length]
            if not codestream.startswith(_SOC):
                raise ValueError("JP2 codestream box does not start with SOC.")
            return codestream
        offset += length
    raise ValueError("JP2 file has no codestream box.")


def _main_header(codestream: bytes, index: int) -> dict:
    if not codestream.startswith(_SOC):
        raise ValueError(f"Frame {index} does not start with a JPEG 2000 SOC marker.")

    offset = 2
    siz = None
    cod = None
    limit = len(codestream)
    while offset + 1 < limit:
        if codestream[offset] != 0xFF:
            raise ValueError(f"Frame {index} JPEG 2000 marker is misaligned.")
        offset += 1
        while offset < limit and codestream[offset] == 0xFF:
            offset += 1
        if offset >= limit:
            break
        marker = codestream[offset]
        offset += 1
        if marker == 0x4F:
            continue
        if marker in _SOT_SOD_EOC:
            break
        if offset + 2 > limit:
            break
        segment_length = int.from_bytes(codestream[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > limit:
            raise ValueError(f"Frame {index} has a truncated JPEG 2000 marker.")
        payload = codestream[offset + 2 : offset + segment_length]
        offset += segment_length
        if marker == 0x51:
            siz = _parse_siz(payload, index)
        elif marker == 0x52:
            cod = _parse_cod(payload, index)

    if siz is None or cod is None:
        raise ValueError(f"Frame {index} is missing a JPEG 2000 SIZ or COD marker.")
    return {**siz, **cod}


def _parse_siz(payload: bytes, index: int) -> dict:
    if len(payload) < 38:
        raise ValueError(f"Frame {index} SIZ marker is too short.")
    width = int.from_bytes(payload[2:6], "big") - int.from_bytes(payload[10:14], "big")
    height = int.from_bytes(payload[6:10], "big") - int.from_bytes(payload[14:18], "big")
    samples = int.from_bytes(payload[34:36], "big")
    if samples < 1 or width < 1 or height < 1:
        raise ValueError(f"Frame {index} has an empty JPEG 2000 image.")
    if len(payload) < 36 + 3 * samples:
        raise ValueError(f"Frame {index} SIZ marker is missing component records.")

    precisions = []
    signed = []
    for component in range(samples):
        start = 36 + 3 * component
        ssiz = payload[start]
        precisions.append((ssiz & 0x7F) + 1)
        signed.append(bool(ssiz & 0x80))
    if len(set(precisions)) != 1 or len(set(signed)) != 1:
        raise ValueError(f"Frame {index} mixes JPEG 2000 component precisions.")

    bits_stored = precisions[0]
    bits_allocated = _bits_allocated(bits_stored)
    return {
        "rows": height,
        "columns": width,
        "samples": samples,
        "bits_stored": bits_stored,
        "bits_allocated": bits_allocated,
        "pixel_representation": 1 if signed[0] else 0,
    }


def _parse_cod(payload: bytes, index: int) -> dict:
    # Scod, progression, layers, multiple-component transform, five SPcod bytes.
    if len(payload) < 10:
        raise ValueError(f"Frame {index} COD marker is too short.")
    mct = payload[4]
    transform = payload[9]
    if mct not in (0, 1) or transform not in (0, 1):
        raise ValueError(f"Frame {index} has an unsupported JPEG 2000 COD marker.")
    return {"mct": bool(mct), "reversible": transform == 1}


def _even_length(codestream: bytes) -> bytes:
    """Return a codestream whose length is even.

    DICOM item values must be even. Padding an odd codestream with a trailing
    ``0x00`` makes Pillow reject the stream, so an odd codestream instead gains
    a one-byte JPEG 2000 comment marker in the main header. The pixels are
    unchanged, and the encapsulated item then needs no trailing pad.
    """
    if len(codestream) % 2 == 0:
        return codestream
    tile = _first_tile_offset(codestream)
    return codestream[:tile] + b"\xff\x64\x00\x03\x00" + codestream[tile:]


def _first_tile_offset(codestream: bytes) -> int:
    offset = 2
    limit = len(codestream)
    while offset + 1 < limit:
        if codestream[offset] != 0xFF:
            break
        start = offset
        offset += 1
        while offset < limit and codestream[offset] == 0xFF:
            offset += 1
        if offset >= limit:
            break
        marker = codestream[offset]
        offset += 1
        if marker in _SOT_SOD_EOC:
            return start
        if offset + 2 > limit:
            break
        segment_length = int.from_bytes(codestream[offset : offset + 2], "big")
        if segment_length < 2:
            break
        offset += segment_length
    raise ValueError("JPEG 2000 codestream has no tile-part marker to pad.")


def _bits_allocated(precision: int) -> int:
    for allocated in (8, 16, 24, 32, 40):
        if precision <= allocated:
            return allocated
    raise ValueError(f"JPEG 2000 precision {precision} is above 40 bits.")
