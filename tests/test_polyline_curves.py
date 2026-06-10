from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.primitives import PolylineCurve


def test_polyline_curve_evaluates_vertices_and_midpoint() -> None:
    curve = PolylineCurve(
        vertices_mm=[
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ]
    )
    points = curve.evaluate([0.0, 3.0 / 7.0, 1.0])

    assert np.allclose(points[0], [0.0, 0.0, 0.0])
    assert np.allclose(points[1], [3.0, 0.0, 0.0])
    assert np.allclose(points[2], [3.0, 4.0, 0.0])
    assert np.isclose(curve.length_mm, 7.0)


def test_polyline_curve_tangents_are_piecewise_unit_vectors() -> None:
    curve = PolylineCurve(
        vertices_mm=[
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ]
    )
    tangents = curve.tangent([0.1, 0.9])

    assert np.allclose(np.linalg.norm(tangents, axis=1), 1.0)
    assert np.allclose(tangents[0], [1.0, 0.0, 0.0])
    assert np.allclose(tangents[1], [0.0, 1.0, 0.0])


def test_polyline_curve_sampling_returns_centreline() -> None:
    curve = PolylineCurve(
        vertices_mm=[
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ]
    )
    centreline = curve.sample(step_mm=2.0, object_id="graph", segment_id="edge")

    assert centreline.object_id == "graph"
    assert centreline.segment_id == "edge"
    assert centreline.n_points >= 5
    assert np.isclose(centreline.length_mm, 7.0)
    assert np.all(np.diff(centreline.arclength_mm) >= 0.0)
    assert any(
        np.allclose(point, [3.0, 0.0, 0.0]) for point in centreline.coordinates_mm
    )


def test_polyline_curve_supports_2d() -> None:
    curve = PolylineCurve(vertices_mm=[[0.0, 0.0], [3.0, 4.0]])
    centreline = curve.sample(n_samples=3)

    assert centreline.ndim == 2
    assert np.isclose(centreline.length_mm, 5.0)


def test_polyline_curve_rejects_zero_length_segment() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        PolylineCurve(
            vertices_mm=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )


def test_polyline_curve_rejects_invalid_parameters() -> None:
    curve = PolylineCurve(vertices_mm=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        curve.evaluate([-0.1, 0.5])


def test_top_level_exports_polyline_curve() -> None:
    import synthworkshop

    assert synthworkshop.PolylineCurve is PolylineCurve
