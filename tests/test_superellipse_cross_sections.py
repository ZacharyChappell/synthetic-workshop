from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.cross_sections import CrossSection, SuperellipseCrossSection


def test_superellipse_cross_section_evaluates_axes_and_corner() -> None:
    cross_section = SuperellipseCrossSection(
        semi_axis_u_mm=4.0,
        semi_axis_v_mm=2.0,
        exponent=4.0,
    )
    result = cross_section.evaluate(
        u_mm=np.array([0.0, 4.0, 0.0, 4.0]),
        v_mm=np.array([0.0, 0.0, 2.0, 2.0]),
    )

    assert np.allclose(result.rho[:3], [0.0, 1.0, 1.0])
    assert result.inside.tolist() == [True, True, True, False]


def test_superellipse_area_matches_ellipse_when_exponent_two() -> None:
    cross_section = SuperellipseCrossSection(
        semi_axis_u_mm=4.0,
        semi_axis_v_mm=2.0,
        exponent=2.0,
    )

    assert np.isclose(cross_section.area_mm2, np.pi * 8.0)


def test_superellipse_boundary_radius_matches_axes() -> None:
    cross_section = SuperellipseCrossSection(
        semi_axis_u_mm=4.0,
        semi_axis_v_mm=2.0,
        exponent=4.0,
    )
    angles = np.array([0.0, np.pi / 2.0, np.pi])

    assert np.allclose(cross_section.boundary_radius_mm(angles), [4.0, 2.0, 4.0])


def test_superellipse_boxier_than_ellipse_at_diagonal() -> None:
    ellipse_like = SuperellipseCrossSection(
        semi_axis_u_mm=4.0,
        semi_axis_v_mm=2.0,
        exponent=2.0,
    )
    boxier = SuperellipseCrossSection(
        semi_axis_u_mm=4.0,
        semi_axis_v_mm=2.0,
        exponent=8.0,
    )
    angle = np.array([np.pi / 4.0])

    assert (
        boxier.boundary_radius_mm(angle)[0] > ellipse_like.boundary_radius_mm(angle)[0]
    )


def test_superellipse_rejects_non_convex_exponent() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        SuperellipseCrossSection(
            semi_axis_u_mm=4.0,
            semi_axis_v_mm=2.0,
            exponent=0.5,
        )


def test_superellipse_satisfies_protocol() -> None:
    cross_section = SuperellipseCrossSection(
        semi_axis_u_mm=4.0,
        semi_axis_v_mm=2.0,
        exponent=4.0,
    )

    assert isinstance(cross_section, CrossSection)


def test_top_level_exports_superellipse() -> None:
    import synthworkshop

    assert synthworkshop.SuperellipseCrossSection is SuperellipseCrossSection
