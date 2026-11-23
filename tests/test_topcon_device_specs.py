"""Conservative Topcon brochure → DICOM optical attribute policy."""

from __future__ import annotations

import pytest

from oct_converter.dicom.fda_meta import (
    apply_topcon_device_specs,
    lookup_topcon_device_specs,
)
from oct_converter.dicom.metadata import OCTImageParams


@pytest.mark.parametrize(
    "model_name,wavelength",
    [
        ("DRI OCT Triton", 1050),
        ("DRI OCT Triton plus", 1050),
        ("3D OCT-2000 FA Plus", 840),
        ("3DOCT-2000FAF", 840),
        ("3D OCT-1000 MARK II", 840),
        ("3DOCT-1000MK2", 840),
    ],
)
def test_wavelength_populated_for_known_models(model_name, wavelength):
    params = OCTImageParams()
    apply_topcon_device_specs(params, model_name)
    assert params.IlluminationWaveLength == wavelength
    assert params.device_specs is True


@pytest.mark.parametrize(
    "model_name",
    [
        "DRI OCT Triton",
        "DRI OCT Triton plus",
        "3D OCT-2000 FA Plus",
        "3D OCT-1000 MARK II",
    ],
)
def test_spatial_resolution_and_bandwidth_not_written_to_dicom_fields(model_name):
    params = OCTImageParams()
    apply_topcon_device_specs(params, model_name)
    assert params.IlluminationBandwidth is None
    assert params.IlluminationPower is None
    assert params.DepthSpatialResolution is None
    assert params.AlongscanSpatialResolution is None
    assert params.AcrossscanSpatialResolution is None
    assert params.MaximumDepthDistortion is None
    assert params.MaximumAlongscanDistortion is None
    assert params.MaximumAcrossscanDistortion is None


def test_published_optical_specs_retained_separately():
    specs = lookup_topcon_device_specs("DRI OCT Triton plus")
    assert specs is not None
    assert specs.wavelength_nm == 1050
    assert specs.optical_axial_resolution_um == 8
    assert specs.optical_lateral_resolution_um == 20
    assert specs.bandwidth_nm is None
    assert specs.dicom_depth_resolution_um is None
    assert specs.dicom_alongscan_resolution_um is None
    assert specs.dicom_acrossscan_resolution_um is None

    sd = lookup_topcon_device_specs("3D OCT-2000 FA Plus")
    assert sd is not None
    assert sd.wavelength_nm == 840
    assert sd.bandwidth_nm == 50  # published half-bandwidth; not yet DICOM-validated
    assert sd.optical_axial_resolution_um == 5.5
    assert sd.optical_lateral_resolution_um == 20
    assert sd.dicom_depth_resolution_um is None


def test_manufacturer_specs_attached_on_image_params():
    params = OCTImageParams()
    apply_topcon_device_specs(params, "3D OCT-1000 MARK II")
    assert params.manufacturer_specs is not None
    assert params.manufacturer_specs.optical_lateral_resolution_um == 20
    assert params.AlongscanSpatialResolution is None


@pytest.mark.parametrize(
    "model_name",
    ["", None, "unknown device", "Camera-2000", "serial 1000 only"],
)
def test_unknown_or_ambiguous_model_leaves_params_unset(model_name):
    params = OCTImageParams()
    apply_topcon_device_specs(params, model_name)
    assert params.IlluminationWaveLength is None
    assert params.device_specs is False
    assert params.manufacturer_specs is None
    assert lookup_topcon_device_specs(model_name) is None


def test_broad_numeric_match_rejected():
    """Do not treat any string containing 1000/2000 as a Topcon OCT."""
    params = OCTImageParams()
    apply_topcon_device_specs(params, "Protocol 2000")
    assert params.IlluminationWaveLength is None
    assert params.device_specs is False
