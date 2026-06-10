from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.cross_sections import (
    CircularCrossSection,
    CrossSection,
    EllipticCrossSection,
    VariableCircularCrossSection,
)


def test_variable_circular_cross_section_radius_at_positions() -> None:
    cross_section = VariableCircularCrossSection(
        radius_start_mm=1.0,
        radius_end_mm=3.0,
        length_mm=10.0,
    )

    radius = cross_section.radius_at(np.array([0.0, 5.0, 10.0, 20.0]))

    assert np.allclose(radius, [1.0, 2.0, 3.0, 3.0])


def test_variable_circular_cross_section_evaluates_longitudinal_radius() -> None:
    cross_section = VariableCircularCrossSection(
        radius_start_mm=1.0,
        radius_end_mm=3.0,
        length_mm=10.0,
    )
    result = cross_section.evaluate(
        u_mm=np.array([1.0, 2.0, 3.0, 3.5]),
        v_mm=np.array([0.0, 0.0, 0.0, 0.0]),
        longitudinal_mm=np.array([0.0, 5.0, 10.0, 10.0]),
    )

    assert np.allclose(result.radius_mm, [1.0, 2.0, 3.0, 3.0])
    assert np.allclose(result.rho[:3], [1.0, 1.0, 1.0])
    assert result.inside.tolist() == [True, True, True, False]


def test_variable_circular_cross_section_requires_longitudinal_coordinate() -> None:
    cross_section = VariableCircularCrossSection(
        radius_start_mm=1.0,
        radius_end_mm=3.0,
        length_mm=10.0,
    )

    with pytest.raises(ValueError, match="longitudinal_mm is required"):
        cross_section.evaluate(
            u_mm=np.array([0.0]),
            v_mm=np.array([0.0]),
        )


def test_variable_circular_cross_section_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="radius_start_mm"):
        VariableCircularCrossSection(
            radius_start_mm=0.0,
            radius_end_mm=3.0,
            length_mm=10.0,
        )

    with pytest.raises(ValueError, match="length_mm"):
        VariableCircularCrossSection(
            radius_start_mm=1.0,
            radius_end_mm=3.0,
            length_mm=0.0,
        )


def test_variable_circular_boundary_radius_uses_maximum_envelope() -> None:
    cross_section = VariableCircularCrossSection(
        radius_start_mm=1.0,
        radius_end_mm=3.0,
        length_mm=10.0,
    )

    radius = cross_section.boundary_radius_mm(np.array([0.0, np.pi / 2.0]))

    assert np.allclose(radius, [3.0, 3.0])


def test_constant_cross_sections_accept_ignored_longitudinal_coordinate() -> None:
    circle = CircularCrossSection(radius_mm=2.0)
    ellipse = EllipticCrossSection(semi_axis_u_mm=3.0, semi_axis_v_mm=1.0)

    circle_result = circle.evaluate(
        u_mm=np.array([2.0]),
        v_mm=np.array([0.0]),
        longitudinal_mm=np.array([10.0]),
    )
    ellipse_result = ellipse.evaluate(
        u_mm=np.array([3.0]),
        v_mm=np.array([0.0]),
        longitudinal_mm=np.array([10.0]),
    )

    assert np.isclose(circle_result.rho[0], 1.0)
    assert np.isclose(ellipse_result.rho[0], 1.0)


def test_variable_circular_cross_section_satisfies_protocol() -> None:
    cross_section = VariableCircularCrossSection(
        radius_start_mm=1.0,
        radius_end_mm=3.0,
        length_mm=10.0,
    )

    assert isinstance(cross_section, CrossSection)


def test_top_level_exports_variable_cross_section() -> None:
    import synthworkshop

    assert synthworkshop.VariableCircularCrossSection is VariableCircularCrossSection
