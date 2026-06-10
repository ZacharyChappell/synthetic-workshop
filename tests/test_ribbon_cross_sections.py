from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.cross_sections import (
    CrossSection,
    RibbonCrossSection,
    SuperellipseCrossSection,
)


def test_ribbon_cross_section_uses_width_and_thickness_as_full_axes() -> None:
    ribbon = RibbonCrossSection(width_mm=10.0, thickness_mm=2.0, exponent=6.0)

    assert ribbon.semi_axes_mm == (5.0, 1.0)
    assert ribbon.semi_axis_u_mm == 5.0
    assert ribbon.semi_axis_v_mm == 1.0


def test_ribbon_cross_section_evaluates_inside_support() -> None:
    ribbon = RibbonCrossSection(width_mm=10.0, thickness_mm=2.0, exponent=6.0)
    result = ribbon.evaluate(
        u_mm=np.array([0.0, 5.0, 0.0, 5.0]),
        v_mm=np.array([0.0, 0.0, 1.0, 1.0]),
    )

    assert np.allclose(result.rho[:3], [0.0, 1.0, 1.0])
    assert result.inside.tolist() == [True, True, True, False]


def test_ribbon_area_matches_equivalent_superellipse() -> None:
    ribbon = RibbonCrossSection(width_mm=10.0, thickness_mm=2.0, exponent=6.0)
    equivalent = SuperellipseCrossSection(
        semi_axis_u_mm=5.0,
        semi_axis_v_mm=1.0,
        exponent=6.0,
    )

    assert np.isclose(ribbon.area_mm2, equivalent.area_mm2)


def test_ribbon_boundary_radius_matches_width_and_thickness_axes() -> None:
    ribbon = RibbonCrossSection(width_mm=10.0, thickness_mm=2.0, exponent=6.0)
    angles = np.array([0.0, np.pi / 2.0])

    assert np.allclose(ribbon.boundary_radius_mm(angles), [5.0, 1.0])


def test_ribbon_rejects_invalid_dimensions_and_exponent() -> None:
    with pytest.raises(ValueError, match="width_mm"):
        RibbonCrossSection(width_mm=0.0, thickness_mm=2.0)

    with pytest.raises(ValueError, match="thickness_mm"):
        RibbonCrossSection(width_mm=10.0, thickness_mm=0.0)

    with pytest.raises(ValueError, match="at least 1"):
        RibbonCrossSection(width_mm=10.0, thickness_mm=2.0, exponent=0.5)


def test_ribbon_satisfies_cross_section_protocol() -> None:
    ribbon = RibbonCrossSection(width_mm=10.0, thickness_mm=2.0, exponent=6.0)

    assert isinstance(ribbon, CrossSection)


def test_top_level_exports_ribbon_cross_section() -> None:
    import synthworkshop

    assert synthworkshop.RibbonCrossSection is RibbonCrossSection
