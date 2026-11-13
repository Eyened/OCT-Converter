from __future__ import annotations

import io
import os
import typing as t
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from construct import ListContainer, StreamError

from oct_converter.image_types import FundusImageWithMetaData, OCTVolumeWithMetaData
from oct_converter.readers.binary_structs import fda_binary
from oct_converter.readers.scan_geometry import topcon_pixel_spacing

def _imdecode(raw: bytes, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Decode a JPEG or JPEG 2000 payload to an array.

    OpenCV releases the GIL while decoding, so callers can decode independent
    slices in parallel. Pillow's decoder does not.
    """
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)
    if image is None:
        raise ValueError("Failed to decode image")
    return image


def _as_gray_if_repeated(image: np.ndarray) -> np.ndarray:
    """Collapse a 3-channel image whose channels are copies of one grayscale.

    Triton OCT angiography B-scans are 8-bit grayscale stored as PNG with
    three identical channels. The OPT image is monochrome.
    """
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        return image
    channels = image[..., :3]
    if np.array_equal(channels[..., 0], channels[..., 1]) and np.array_equal(
        channels[..., 1], channels[..., 2]
    ):
        return np.ascontiguousarray(channels[..., 0])
    return image


def _decode_slices(raw_slices: list[bytes]) -> list[np.ndarray]:
    if len(raw_slices) <= 1:
        decoded = [_imdecode(raw) for raw in raw_slices]
    else:
        workers = min(len(raw_slices), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            decoded = list(pool.map(_imdecode, raw_slices))
    return [_as_gray_if_repeated(image) for image in decoded]


def _read_chunk_bytes(f, chunk_location, chunk_size) -> bytes:
    """Read one chunk payload. Repeated chunks use the first occurrence."""
    if isinstance(chunk_location, list):
        f.seek(chunk_location[0])
        return f.read(chunk_size[0])
    f.seek(chunk_location)
    return f.read(chunk_size)


class FDA(object):
    """Class for extracting data from Topcon's .fda file format.

    Attributes:
        filepath: path to .fda file for reading.
        chunk_dict: names of data chunks present in the file, and their start locations.
    """

    def __init__(self, filepath: str | Path, printing: bool = False) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(self.filepath)

        self.chunk_dict, self.header = self.get_list_of_file_chunks(printing=printing)

    def get_list_of_file_chunks(self, printing: bool = False) -> t.Tuple[dict, dict]:
        """Find all data chunks present in the file.

        Returns:
            chunk_dict: dictionary of chunk names, containing their locations
            in the file and size.
            header: dictionary of file header information
        """
        chunk_dict = {}
        with open(self.filepath, "rb") as f:
            raw = f.read(15)
            header = dict(fda_binary.header.parse(raw))

            eof = False
            while not eof:
                chunk_name_size = np.frombuffer(f.read(1), dtype=np.uint8)[0]
                if chunk_name_size == 0:
                    eof = True
                else:
                    chunk_name = f.read(chunk_name_size)
                    chunk_size = np.frombuffer(f.read(4), dtype=np.uint32)[0]
                    chunk_location = f.tell()
                    f.seek(chunk_size, 1)
                    if chunk_name in chunk_dict.keys():
                        previous_location = chunk_dict[chunk_name][0]
                        previous_size = chunk_dict[chunk_name][1]
                        if not isinstance(previous_location, list):
                            previous_location = [previous_location]
                            previous_size = [previous_size]
                        chunk_location = previous_location + [chunk_location]
                        chunk_size = previous_size + [chunk_size]

                    chunk_dict[chunk_name] = [chunk_location, chunk_size]
        if printing:
            print("File {} contains the following chunks:".format(self.filepath))
            for key in chunk_dict.keys():
                print(key)
            print("")
        return chunk_dict, header

    def read_oct_volume(
        self, verbose=False, *, decode: bool = True
    ) -> OCTVolumeWithMetaData:
        """Reads OCT data.

        Notes:
            Mostly based on description of .fda file format here:
            https://bitbucket.org/uocte/uocte/wiki/Topcon%20File%20Format

        Args:
            verbose: if True, prints unsupported metadata chunks.
            decode: if False and the volume is ``@IMG_JPEG``, ``volume`` is
                the raw JPEG 2000 payloads instead of decoded arrays.

        Returns:
            OCTVolumeWithMetaData
        """
        # read oct data chunk, whether that's IMG_JPEG or IMG_MOT_COMP_03
        # TODO: could support the other IMG_MOT_COMP variants as well.
        volume, oct_header = self.read_oct_data_chunk(decode=decode)
        # if oct data is found, calculate pixel spacing.
        if oct_header:
            pixel_spacing = self.read_scan_params(oct_header)
        else:
            pixel_spacing = None

        # read segmentation contours if possible and store them as distance
        # from top of scan to be compatible with plotting in OCTVolume
        contours = self.read_segmentation()
        if contours:
            contours = {k: oct_header.get("height") - v for k, v in contours.items()}

        # read all other metadata
        metadata = self.read_all_metadata(verbose=verbose)
        patient_info = metadata.get("patient_info_02") or metadata.get(
            "patient_info", {}
        )
        capture_info = metadata.get("capture_info_02") or metadata.get(
            "capture_info", {}
        )
        sex_map = {1: "M", 2: "F", 3: "O", None: ""}
        lat_map = {0: "R", 1: "L", None: ""}

        try:
            patient_dob = datetime(*patient_info.get("birth_date"))
        except (TypeError, ValueError):
            patient_dob = None
        oct_volume = OCTVolumeWithMetaData(
            volume,
            patient_id=patient_info.get("patient_id"),
            first_name=patient_info.get("first_name"),
            surname=patient_info.get("last_name"),
            sex=sex_map[patient_info.get("sex", None)],
            patient_dob=patient_dob,
            acquisition_date=datetime(*capture_info.get("cap_date")),
            laterality=lat_map[capture_info.get("eye", None)],
            contours=contours,
            pixel_spacing=pixel_spacing,
            metadata=metadata,
            header=self.header,
            oct_header=oct_header,
        )

        return oct_volume

    def read_oct_data_chunk(self, *, decode: bool = True) -> t.Tuple[list, dict]:
        """Given available chunks, identifies which chunk to utilize
        as the primary OCT data and returns the volume.

        Args:
            decode: if False, ``@IMG_JPEG`` slices are returned as the raw
                bytes stored in the file (JPEG 2000) instead of arrays.

        Returns:
            volume: OCT volume as a list of arrays, or JPEG 2000 payloads
            oct_header: OCT header as dict
        """
        if b"@IMG_JPEG" in self.chunk_dict:
            with open(self.filepath, "rb") as f:
                chunk_location, chunk_size = self.chunk_dict[b"@IMG_JPEG"]
                f.seek(chunk_location)  # Set the chunk’s current position.
                raw = f.read(25)
                oct_header = fda_binary.oct_header.parse(raw)

                raw_slices = []
                for _ in range(oct_header.number_slices):
                    size = int.from_bytes(f.read(4), "little", signed=True)
                    raw_slices.append(f.read(size))
            if not decode:
                return raw_slices, dict(oct_header)
            volume = _decode_slices(raw_slices)
            return volume, dict(oct_header)

        elif b"@IMG_MOT_COMP_03" in self.chunk_dict:
            with open(self.filepath, "rb") as f:
                chunk_location, chunk_size = self.chunk_dict[b"@IMG_MOT_COMP_03"]
                f.seek(chunk_location)  # Set the chunk’s current position.
                raw = f.read(22)
                oct_header = fda_binary.oct_header_2.parse(raw)
                number_pixels = (
                    oct_header.width * oct_header.height * oct_header.number_slices
                )
                raw_volume = np.frombuffer(f.read(number_pixels * 2), dtype=np.uint16)
                volume = np.array(raw_volume)
                volume = volume.reshape(
                    oct_header.width,
                    oct_header.height,
                    oct_header.number_slices,
                    order="F",
                )
                volume = np.transpose(volume, [1, 0, 2])
                adjusted_volume = [volume[:, :, i] for i in range(volume.shape[2])]
            return adjusted_volume, dict(oct_header)

        else:
            print(
                "Neither @IMG_JPEG nor @IMG_MOT_COMP_03 found. OCT volume not identified."
            )
            return None, None

    def read_scan_params(self, oct_header: dict) -> list:
        """Given available chunks, identifies available PARAM_SCAN chunk
        and calculates pixel spacing.

        Args:
            oct_header: OCT header information as dict

        Returns:
            pixel_spacing: list of pixel spacing, ordered by width, slice thickness, height.
        """
        if (
            b"@PARAM_SCAN_04" not in self.chunk_dict
            and b"@PARAM_SCAN_02" not in self.chunk_dict
        ):
            print(
                "Neither @PARAM_SCAN_04 nor @PARAM_SCAN_02 found. Pixel spacing not calculated."
            )
            return None
        with open(self.filepath, "rb") as f:
            if b"@PARAM_SCAN_04" in self.chunk_dict:
                chunk_loc, chunk_size = self.chunk_dict.get(
                    b"@PARAM_SCAN_04", (None, None)
                )
                f.seek(chunk_loc)
                scan_params = fda_binary.param_scan_04_header.parse(f.read(chunk_size))
            elif b"@PARAM_SCAN_02" in self.chunk_dict:
                chunk_loc, chunk_size = self.chunk_dict.get(
                    b"@PARAM_SCAN_02", (None, None)
                )
                f.seek(chunk_loc)
                scan_params = fda_binary.param_scan_02_header.parse(f.read(chunk_size))
            x_mm = float(scan_params.get("x_dimension_mm") or 0)
            y_mm = float(scan_params.get("y_dimension_mm") or 0)
            oct_header["x_dimension_mm"] = x_mm
            oct_header["y_dimension_mm"] = y_mm
            pixel_spacing = topcon_pixel_spacing(
                oct_header.get("scan_mode"),
                oct_header.get("width"),
                oct_header.get("number_slices"),
                x_mm,
                y_mm,
                scan_params.get("z_resolution_um") or 0,
            )
        return pixel_spacing

    def read_fundus_image(self) -> FundusImageWithMetaData:
        """Reads fundus image.

        Returns:
            FundusImageWithMetaData
        """
        if b"@IMG_FUNDUS" not in self.chunk_dict:
            print("@IMG_FUNDUS is not in chunk list, skipping.")
            return None
        with open(self.filepath, "rb") as f:
            chunk_location, chunk_size = self.chunk_dict[b"@IMG_FUNDUS"]
            f.seek(chunk_location)  # Set the chunk’s current position.
            raw = f.read(24)  # skip 24 is important
            fundus_header = fda_binary.fundus_header.parse(raw)
            raw_image = f.read(fundus_header.size)
            # cv2 returns BGR, which matches the previous Pillow path
            # (RGB decoded, then a channel-axis flip).
            image = _imdecode(raw_image)
        fundus_image = FundusImageWithMetaData(image)
        return fundus_image

    def read_fundus_image_bytes(self) -> bytes | None:
        """Return the ``@IMG_FUNDUS`` payload without decoding it."""
        return self._read_sized_image(
            b"@IMG_FUNDUS", fda_binary.fundus_header, 24
        )

    def read_fundus_image_gray_scale(self) -> FundusImageWithMetaData:
        """Reads grayscale fundus image.

        Returns:
            FundusImageWithMetaData
        """
        if b"@IMG_TRC_02" not in self.chunk_dict:
            print("@IMG_TRC_02 is not in chunk list, skipping.")
            return None
        with open(self.filepath, "rb") as f:
            chunk_location, chunk_size = self.chunk_dict[b"@IMG_TRC_02"]
            f.seek(chunk_location)  # Set the chunk’s current position.
            raw = f.read(21)  # skip 21 is important
            img_trc_02_header = fda_binary.img_trc_02_header.parse(raw)
            raw_image = f.read(img_trc_02_header.size)
            # TRC JPEGs are often encoded as RGB with identical channels;
            # force true 2-D grayscale for DICOM MONOCHROME2 export.
            image = _imdecode(raw_image, flags=cv2.IMREAD_GRAYSCALE)
        fundus_gray_scale_image = FundusImageWithMetaData(image)
        return fundus_gray_scale_image

    def read_fundus_image_gray_scale_bytes(self) -> bytes | None:
        """Return the ``@IMG_TRC_02`` payload without decoding it."""
        return self._read_sized_image(
            b"@IMG_TRC_02", fda_binary.img_trc_02_header, 21
        )

    def _read_sized_image(
        self, chunk_name: bytes, header, header_size: int
    ) -> bytes | None:
        """Read one image chunk's payload using the size field in its header."""
        if chunk_name not in self.chunk_dict:
            return None
        with open(self.filepath, "rb") as f:
            chunk_location, _chunk_size = self.chunk_dict[chunk_name]
            if isinstance(chunk_location, list):
                chunk_location = chunk_location[0]
            f.seek(chunk_location)
            parsed = header.parse(f.read(header_size))
            return f.read(int(parsed.size))

    def read_segmentation(self) -> dict:
        """Reads layer segmentation data.

        Segmentation values are returned in a dictionary with a key for every
        layer boundary and are measured in pixels from B-scan bottom.

        Returns
            dictionary with segmentation data.
        """

        if b"@CONTOUR_INFO" not in self.chunk_dict.keys():
            print("The file does not have any segmentation chunk.")
            return None

        layer = {
            "MULTILAYERS_1": "ILM",
            "MULTILAYERS_2": "RNFL_GCL",
            "MULTILAYERS_3": "GCL_IPL",
            "MULTILAYERS_4": "IPL_INL",
            "MULTILAYERS_5": "MZ_EZ",
            "MULTILAYERS_6": "IZ_RPE",
            "MULTILAYERS_7": "BM",
            "MULTILAYERS_8": "INL_OPL",
            "MULTILAYERS_9": "ELM",
            "MULTILAYERS_10": "CSI",
        }

        seg_dict = {}
        with open(self.filepath, "rb") as f:
            for pos in self.chunk_dict[b"@CONTOUR_INFO"][0]:
                f.seek(pos)
                raw = f.read(34)
                header = dict(fda_binary.__dict__["contour_info_header"].parse(raw))
                n_voxel = header["width"] * header["height"]

                data_bytes = f.read(n_voxel * 2)
                seg = np.frombuffer(data_bytes, dtype="<u2").reshape(
                    (header["height"], header["width"])
                )
                seg = np.flip(seg, 0)
                layer_name = layer.get(header["id"], header["id"])
                seg_dict[layer_name] = seg

        return seg_dict

    def read_all_metadata(self, verbose: bool = False):
        """
        Reads all available metadata and returns a dictionary.

        Args:
            verbose: if True, prints the chunks that are not supported.

        Returns:
            dictionary with all metadata.
        """
        metadata = dict()
        for key in self.chunk_dict.keys():
            if key in [b"@IMG_JPEG", b"@IMG_FUNDUS", b"@IMG_TRC_02", b"@CONTOUR_INFO"]:
                # these chunks have their own dedicated methods for extraction
                continue
            elif key == b"@PARAM_OBS_02":
                metadata["param_obs_02"] = self.read_param_obs()
                continue
            json_key = key.decode().split("@")[-1].lower()
            try:
                metadata[json_key] = self.read_any_info_and_make_dict(key)
            except KeyError:
                if verbose:
                    print(f"{key} there is no method for getting info from this chunk.")
        return metadata

    def read_any_info_and_make_dict(self, chunk_name: str) -> dict:
        """Reads any chunk and constructs a dictionary.

        Args:
            chunk_name: name of the chunk which data will be taken.

        Returns:
            Chunk info data
        """
        if chunk_name not in self.chunk_dict:
            print(f"{chunk_name} is not in chunk list, skipping.")
            return None
        with open(self.filepath, "rb") as f:
            chunk_location, chunk_size = self.chunk_dict[chunk_name]
            start = (
                chunk_location[0] if isinstance(chunk_location, list) else chunk_location
            )
            raw = _read_chunk_bytes(f, chunk_location, chunk_size)
            header_name = f"{chunk_name.decode().split('@')[-1].lower()}_header"
            parser = fda_binary.__dict__[header_name]
            try:
                chunk_info_header = dict(parser.parse(raw))
            except StreamError:
                # A few structs (ALIGN_INFO, PATIENTEXT_INFO) declare more
                # bytes than the chunk length and historically parsed by
                # reading into the following chunk.
                f.seek(start)
                chunk_info_header = dict(parser.parse(f.read()))
            chunks_info = dict()
            for idx, key in enumerate(chunk_info_header.keys()):
                if idx == 0:
                    continue
                if type(chunk_info_header[key]) is ListContainer:
                    chunks_info[key] = list(chunk_info_header[key])
                else:
                    chunks_info[key] = chunk_info_header[key]
        return chunks_info

    def read_param_obs(self) -> dict:
        """Reads PARAM_OBS_02 while accounting for varied chunk sizes.

        Returns:
            Chunk info data for PARAM_OBS_02
        """
        with open(self.filepath, "rb") as f:
            chunk_location, chunk_size = self.chunk_dict[b"@PARAM_OBS_02"]
            raw = _read_chunk_bytes(f, chunk_location, chunk_size)
            # PARAM_OBS_02 is either of size 90 or size 6.
            if chunk_size == 90:
                chunk_info_header = dict(fda_binary.param_obs_02_header.parse(raw))
            else:  # chunk_size == 6
                chunk_info_header = dict(
                    fda_binary.param_obs_02_short_header.parse(raw)
                )

            chunks_info = dict()
            for idx, key in enumerate(chunk_info_header.keys()):
                if idx == 0:
                    continue
                if type(chunk_info_header[key]) is ListContainer:
                    chunks_info[key] = list(chunk_info_header[key])
                else:
                    chunks_info[key] = chunk_info_header[key]
        return chunks_info
