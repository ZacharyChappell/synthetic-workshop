from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.profiles import (
    MultiPeakRadialProfile,
    OneSidedLesionProfile,
    PeriodicLongitudinalProfile,
    RadialLongitudinalGradientProfile,
    ScalarProfile,
)


def test_multi_peak_radial_profile_has_two_radial_maxima() -> None:
    profile = MultiPeakRadialProfile(
        base_value=0.0,
        peak_centres_fraction=(0.25, 0.75),
        peak_amplitudes=(1.0, 0.5),
        peak_widths_fraction=(0.03, 0.03),
        background_value=-1.0,
    )

    result = profile.evaluate(
        rho=np.array([0.25, 0.50, 0.75, 1.5]),
        radial_distance_mm=np.array([0.25, 0.50, 0.75, 1.5]),
        inside=np.array([True, True, True, False]),
    )

    assert result.values[0] > result.values[1]
    assert result.values[2] > result.values[1]
    assert np.isclose(result.values[3], -1.0)


def test_multi_peak_radial_profile_rejects_mismatched_peak_lengths() -> None:
    with pytest.raises(ValueError, match="matching lengths"):
        MultiPeakRadialProfile(
            peak_centres_fraction=(0.25, 0.75),
            peak_amplitudes=(1.0,),
            peak_widths_fraction=(0.05, 0.05),
        )


def test_one_sided_lesion_profile_is_asymmetric() -> None:
    profile = OneSidedLesionProfile(
        baseline_value=0.2,
        lesion_delta=1.0,
        lesion_side="positive",
        lesion_centre_mm=1.0,
        lesion_width_mm=0.2,
        background_value=-1.0,
    )

    result = profile.evaluate(
        rho=np.array([0.5, 0.5, 0.5]),
        radial_distance_mm=np.array([1.0, 1.0, 1.0]),
        signed_u_mm=np.array([-1.0, 1.0, 2.0]),
        inside=np.array([True, True, False]),
    )

    assert np.isclose(result.values[0], 0.2)
    assert result.values[1] > 1.0
    assert np.isclose(result.values[2], -1.0)


def test_one_sided_lesion_profile_requires_signed_u_coordinate() -> None:
    profile = OneSidedLesionProfile()

    with pytest.raises(ValueError, match="signed_u_mm is required"):
        profile.evaluate(
            rho=np.array([0.5]),
            radial_distance_mm=np.array([1.0]),
        )


def test_radial_longitudinal_gradient_profile_combines_both_axes() -> None:
    profile = RadialLongitudinalGradientProfile(
        centre_start_value=1.0,
        centre_end_value=2.0,
        edge_start_value=0.0,
        edge_end_value=1.0,
        length_mm=10.0,
        background_value=-1.0,
    )

    result = profile.evaluate(
        rho=np.array([0.0, 1.0, 0.0, 1.0, 0.5]),
        radial_distance_mm=np.array([0.0, 1.0, 0.0, 1.0, 0.5]),
        longitudinal_mm=np.array([0.0, 0.0, 10.0, 10.0, 5.0]),
        inside=np.array([True, True, True, True, False]),
    )

    assert np.allclose(result.values, [1.0, 0.0, 2.0, 1.0, -1.0])


def test_radial_longitudinal_gradient_requires_longitudinal_coordinate() -> None:
    profile = RadialLongitudinalGradientProfile(length_mm=10.0)

    with pytest.raises(ValueError, match="longitudinal_mm is required"):
        profile.evaluate(
            rho=np.array([0.0]),
            radial_distance_mm=np.array([0.0]),
        )


def test_periodic_longitudinal_profile_oscillates_along_length() -> None:
    profile = PeriodicLongitudinalProfile(
        baseline_value=1.0,
        amplitude=0.5,
        length_mm=10.0,
        periods=1.0,
        phase_radians=0.0,
        background_value=-1.0,
    )

    result = profile.evaluate(
        rho=np.zeros(5),
        radial_distance_mm=np.zeros(5),
        longitudinal_mm=np.array([0.0, 2.5, 5.0, 7.5, 5.0]),
        inside=np.array([True, True, True, True, False]),
    )

    assert np.allclose(result.values[:4], [1.0, 1.5, 1.0, 0.5])
    assert np.isclose(result.values[4], -1.0)


def test_periodic_longitudinal_profile_rejects_invalid_periods() -> None:
    with pytest.raises(ValueError, match="periods"):
        PeriodicLongitudinalProfile(periods=0.0)


def test_new_m2fb_profiles_satisfy_protocol() -> None:
    assert isinstance(MultiPeakRadialProfile(), ScalarProfile)
    assert isinstance(OneSidedLesionProfile(), ScalarProfile)
    assert isinstance(RadialLongitudinalGradientProfile(), ScalarProfile)
    assert isinstance(PeriodicLongitudinalProfile(), ScalarProfile)


def test_top_level_exports_m2fb_profiles() -> None:
    import synthworkshop

    assert synthworkshop.MultiPeakRadialProfile is MultiPeakRadialProfile
    assert synthworkshop.OneSidedLesionProfile is OneSidedLesionProfile
    assert (
        synthworkshop.RadialLongitudinalGradientProfile
        is RadialLongitudinalGradientProfile
    )
    assert synthworkshop.PeriodicLongitudinalProfile is PeriodicLongitudinalProfile
