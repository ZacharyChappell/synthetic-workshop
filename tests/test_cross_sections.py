from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.cross_sections import (
    CircularCrossSection,
    CrossSection,
    CrossSectionEvaluation,
    EllipticCrossSection,
)
from synthworkshop.cross_sections.base import validate_local_coordinates


def test_circular_cross_section_evaluates_radius() -> None:
    cross_section = CircularCrossSection(radius_mm=5.0)
    result = cross_section.evaluate(
        u_mm=np.array([0.0, 3.0, 6.0]),
        v_mm=np.array([0.0, 4.0, 0.0]),
    )

    assert np.allclose(result.radial_distance_mm, [0.0, 5.0, 6.0])
    assert np.allclose(result.rho, [0.0, 1.0, 1.2])
    assert result.inside.tolist() == [True, True, False]


def test_circular_cross_section_area_and_boundary_radius() -> None:
    cross_section = CircularCrossSection(radius_mm=2.0)
    angles = np.linspace(0.0, 2.0 * np.pi, 8)

    assert np.isclose(cross_section.area_mm2, np.pi * 4.0)
    assert np.allclose(cross_section.boundary_radius_mm(angles), 2.0)
    assert cross_section.semi_axes_mm == (2.0, 2.0)


def test_circular_cross_section_rejects_invalid_radius() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        CircularCrossSection(radius_mm=0.0)


def test_elliptic_cross_section_evaluates_axes() -> None:
    cross_section = EllipticCrossSection(
        semi_axis_u_mm=4.0,
        semi_axis_v_mm=2.0,
    )
    result = cross_section.evaluate(
        u_mm=np.array([0.0, 4.0, 0.0, 0.0]),
        v_mm=np.array([0.0, 0.0, 2.0, 2.1]),
    )

    assert np.allclose(result.rho[:3], [0.0, 1.0, 1.0])
    assert result.inside.tolist() == [True, True, True, False]


def test_elliptic_boundary_radius_matches_axes() -> None:
    cross_section = EllipticCrossSection(
        semi_axis_u_mm=4.0,
        semi_axis_v_mm=2.0,
    )
    angles = np.array([0.0, np.pi / 2.0, np.pi])

    assert np.allclose(cross_section.boundary_radius_mm(angles), [4.0, 2.0, 4.0])
    assert np.isclose(cross_section.area_mm2, np.pi * 8.0)
    assert cross_section.semi_axes_mm == (4.0, 2.0)


def test_elliptic_cross_section_rejects_invalid_axes() -> None:
    with pytest.raises(ValueError, match="semi_axis_u_mm"):
        EllipticCrossSection(semi_axis_u_mm=-1.0, semi_axis_v_mm=2.0)

    with pytest.raises(ValueError, match="semi_axis_v_mm"):
        EllipticCrossSection(semi_axis_u_mm=1.0, semi_axis_v_mm=0.0)


def test_cross_section_evaluation_accepts_grid_shaped_arrays() -> None:
    cross_section = CircularCrossSection(radius_mm=2.0)
    u, v = np.meshgrid(
        np.linspace(-2.0, 2.0, 5),
        np.linspace(-2.0, 2.0, 5),
        indexing="ij",
    )
    result = cross_section.evaluate(u, v)

    assert result.shape == (5, 5)
    assert result.n_points == 25
    assert result.inside_fraction > 0.0
    assert result.inside_fraction < 1.0


def test_cross_section_evaluation_to_dataframe_flattens_values() -> None:
    result = CircularCrossSection(radius_mm=1.0).evaluate(
        u_mm=np.array([[0.0, 1.0]]),
        v_mm=np.array([[0.0, 0.0]]),
    )
    table = result.to_dataframe()

    assert table.shape[0] == 2
    assert {"u_mm", "v_mm", "rho", "radial_distance_mm", "inside"}.issubset(
        table.columns
    )


def test_validate_local_coordinates_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        validate_local_coordinates([0.0, 1.0], [[0.0, 1.0]])


def test_cross_section_evaluation_rejects_negative_rho() -> None:
    with pytest.raises(ValueError, match="rho must be non-negative"):
        CrossSectionEvaluation(
            u_mm=[0.0],
            v_mm=[0.0],
            rho=[-1.0],
            radial_distance_mm=[0.0],
            inside=[True],
        )


def test_cross_sections_satisfy_protocol() -> None:
    circle = CircularCrossSection(radius_mm=1.0)
    ellipse = EllipticCrossSection(semi_axis_u_mm=2.0, semi_axis_v_mm=1.0)

    assert isinstance(circle, CrossSection)
    assert isinstance(ellipse, CrossSection)


def test_top_level_exports_cross_sections() -> None:
    import synthworkshop

    assert synthworkshop.CircularCrossSection is CircularCrossSection
    assert synthworkshop.EllipticCrossSection is EllipticCrossSection
