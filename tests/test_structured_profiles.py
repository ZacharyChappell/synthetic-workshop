from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.profiles import (
    HollowCoreProfile,
    LongitudinalGradientProfile,
    ScalarProfile,
    SigmoidBoundaryProfile,
)


def test_hollow_core_profile_has_low_core_bright_shell_and_background() -> None:
    profile = HollowCoreProfile(
        core_value=0.1,
        shell_value=1.0,
        edge_value=0.2,
        core_radius_fraction=0.25,
        shell_radius_fraction=0.75,
        background_value=-1.0,
    )

    result = profile.evaluate(
        rho=np.array([0.0, 0.25, 0.75, 1.0, 1.5]),
        radial_distance_mm=np.array([0.0, 0.5, 1.5, 2.0, 3.0]),
        inside=np.array([True, True, True, True, False]),
    )

    assert np.allclose(result.values, [0.1, 0.1, 1.0, 0.2, -1.0])


def test_hollow_core_profile_rejects_invalid_shell_order() -> None:
    with pytest.raises(ValueError, match="shell_radius_fraction"):
        HollowCoreProfile(
            core_radius_fraction=0.6,
            shell_radius_fraction=0.5,
        )


def test_sigmoid_boundary_profile_decreases_from_centre_to_edge() -> None:
    profile = SigmoidBoundaryProfile(
        centre_value=1.0,
        edge_value=0.2,
        boundary_fraction=0.5,
        width_fraction=0.05,
        background_value=-1.0,
    )

    result = profile.evaluate(
        rho=np.array([0.0, 0.5, 1.0, 1.5]),
        radial_distance_mm=np.array([0.0, 1.0, 2.0, 3.0]),
        inside=np.array([True, True, True, False]),
    )

    assert result.values[0] > result.values[1] > result.values[2]
    assert np.isclose(result.values[1], 0.6, atol=1e-6)
    assert np.isclose(result.values[3], -1.0)


def test_sigmoid_boundary_profile_rejects_invalid_width() -> None:
    with pytest.raises(ValueError, match="width_fraction"):
        SigmoidBoundaryProfile(width_fraction=0.0)


def test_longitudinal_gradient_profile_interpolates_start_to_end() -> None:
    profile = LongitudinalGradientProfile(
        start_value=0.2,
        end_value=1.0,
        length_mm=10.0,
        background_value=-1.0,
    )

    result = profile.evaluate(
        rho=np.array([0.0, 0.0, 0.0, 0.0, 1.5]),
        radial_distance_mm=np.array([0.0, 0.0, 0.0, 0.0, 3.0]),
        longitudinal_mm=np.array([0.0, 5.0, 10.0, 15.0, 5.0]),
        inside=np.array([True, True, True, True, False]),
    )

    assert np.allclose(result.values, [0.2, 0.6, 1.0, 1.0, -1.0])


def test_longitudinal_gradient_profile_requires_longitudinal_coordinate() -> None:
    profile = LongitudinalGradientProfile(length_mm=10.0)

    with pytest.raises(ValueError, match="longitudinal_mm is required"):
        profile.evaluate(
            rho=np.array([0.0]),
            radial_distance_mm=np.array([0.0]),
        )


def test_new_structured_profiles_satisfy_protocol() -> None:
    assert isinstance(HollowCoreProfile(), ScalarProfile)
    assert isinstance(SigmoidBoundaryProfile(), ScalarProfile)
    assert isinstance(LongitudinalGradientProfile(), ScalarProfile)


def test_top_level_exports_structured_profiles() -> None:
    import synthworkshop

    assert synthworkshop.HollowCoreProfile is HollowCoreProfile
    assert synthworkshop.SigmoidBoundaryProfile is SigmoidBoundaryProfile
    assert synthworkshop.LongitudinalGradientProfile is LongitudinalGradientProfile
