from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.profiles import (
    AsymmetricLinearProfile,
    ConstantProfile,
    EdgeEnhancedProfile,
    GaussianRadialProfile,
    LinearRadialProfile,
    ScalarProfile,
)


def test_constant_profile_sets_inside_and_background() -> None:
    profile = ConstantProfile(value=2.0, background_value=-1.0)
    result = profile.evaluate(
        rho=np.array([0.0, 0.5, 1.5]),
        radial_distance_mm=np.array([0.0, 1.0, 3.0]),
        inside=np.array([True, True, False]),
    )

    assert np.allclose(result.values, [2.0, 2.0, -1.0])


def test_linear_radial_profile_interpolates_centre_to_edge() -> None:
    profile = LinearRadialProfile(
        centre_value=1.0,
        edge_value=0.2,
        background_value=0.0,
    )
    result = profile.evaluate(
        rho=np.array([0.0, 0.5, 1.0, 1.5]),
        radial_distance_mm=np.array([0.0, 1.0, 2.0, 3.0]),
        inside=np.array([True, True, True, False]),
    )

    assert np.allclose(result.values, [1.0, 0.6, 0.2, 0.0])


def test_gaussian_radial_profile_is_centre_bright() -> None:
    profile = GaussianRadialProfile(
        centre_value=1.0,
        edge_value=0.2,
        sigma_fraction=0.5,
    )
    result = profile.evaluate(
        rho=np.array([0.0, 0.5, 1.0]),
        radial_distance_mm=np.array([0.0, 1.0, 2.0]),
    )

    assert result.values[0] > result.values[1]
    assert result.values[1] > result.values[2]


def test_edge_enhanced_profile_peaks_near_boundary() -> None:
    profile = EdgeEnhancedProfile(
        centre_value=0.2,
        edge_value=1.0,
        edge_width_fraction=0.2,
    )
    result = profile.evaluate(
        rho=np.array([0.0, 0.5, 1.0]),
        radial_distance_mm=np.array([0.0, 1.0, 2.0]),
    )

    assert result.values[2] > result.values[1]
    assert result.values[1] > result.values[0]


def test_asymmetric_profile_requires_signed_u() -> None:
    profile = AsymmetricLinearProfile()

    with pytest.raises(ValueError, match="signed_u_mm is required"):
        profile.evaluate(
            rho=np.array([0.0]),
            radial_distance_mm=np.array([0.0]),
        )


def test_asymmetric_profile_separates_signed_sides() -> None:
    profile = AsymmetricLinearProfile(
        centre_value=1.0,
        edge_value=0.2,
        asymmetry=0.1,
    )
    result = profile.evaluate(
        rho=np.array([1.0, 1.0, 0.0]),
        radial_distance_mm=np.array([2.0, 2.0, 0.0]),
        signed_u_mm=np.array([-2.0, 2.0, 0.0]),
    )

    assert result.values[0] < result.values[1]
    assert np.isclose(result.values[2], 1.0)


def test_profile_protocol_runtime_check() -> None:
    assert isinstance(ConstantProfile(), ScalarProfile)
    assert isinstance(LinearRadialProfile(), ScalarProfile)


def test_profile_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="centre_value"):
        LinearRadialProfile(centre_value=np.nan)
