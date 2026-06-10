from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.primitives.curves import (
    Centreline,
    LineCurve,
    SinusoidalCurve,
    cumulative_arclength_mm,
)


def test_line_curve_evaluates_endpoints() -> None:
    curve = LineCurve(start_mm=(0.0, 0.0, 0.0), end_mm=(10.0, 0.0, 0.0))
    points = curve.evaluate([0.0, 0.5, 1.0])

    assert np.allclose(points[0], [0.0, 0.0, 0.0])
    assert np.allclose(points[1], [5.0, 0.0, 0.0])
    assert np.allclose(points[2], [10.0, 0.0, 0.0])


def test_line_curve_tangents_are_unit_and_constant() -> None:
    curve = LineCurve(start_mm=(0.0, 0.0, 0.0), end_mm=(0.0, 3.0, 4.0))
    tangents = curve.tangent([0.0, 0.5, 1.0])

    assert np.allclose(np.linalg.norm(tangents, axis=1), 1.0)
    assert np.allclose(tangents, tangents[0])


def test_line_curve_sampling_uses_endpoint() -> None:
    curve = LineCurve(start_mm=(0.0, 0.0, 0.0), end_mm=(10.0, 0.0, 0.0))
    centreline = curve.sample(step_mm=2.0, object_id="target")

    assert centreline.n_points == 6
    assert centreline.object_id == "target"
    assert np.allclose(centreline.coordinates_mm[0], [0.0, 0.0, 0.0])
    assert np.allclose(centreline.coordinates_mm[-1], [10.0, 0.0, 0.0])
    assert np.isclose(centreline.length_mm, 10.0)


def test_line_curve_supports_2d() -> None:
    curve = LineCurve(start_mm=(0.0, 0.0), end_mm=(3.0, 4.0))
    centreline = curve.sample(n_samples=3)

    assert centreline.ndim == 2
    assert np.isclose(centreline.length_mm, 5.0)


def test_line_curve_rejects_identical_points() -> None:
    with pytest.raises(ValueError, match="must differ"):
        LineCurve(start_mm=(1.0, 1.0, 1.0), end_mm=(1.0, 1.0, 1.0))


def test_sinusoidal_curve_preserves_endpoints() -> None:
    curve = SinusoidalCurve(
        start_mm=(0.0, 0.0, 0.0),
        end_mm=(10.0, 0.0, 0.0),
        amplitude_mm=(0.0, 2.0, 0.0),
        periods=1.0,
    )
    points = curve.evaluate([0.0, 0.25, 0.5, 0.75, 1.0])

    assert np.allclose(points[0], [0.0, 0.0, 0.0])
    assert np.allclose(points[-1], [10.0, 0.0, 0.0])
    assert np.max(np.abs(points[:, 1])) > 0.5


def test_sinusoidal_curve_tangents_are_unit_vectors() -> None:
    curve = SinusoidalCurve(
        start_mm=(0.0, 0.0, 0.0),
        end_mm=(10.0, 0.0, 0.0),
        amplitude_mm=(0.0, 2.0, 0.0),
        periods=1.0,
    )
    tangents = curve.tangent(np.linspace(0.0, 1.0, 9))

    assert np.allclose(np.linalg.norm(tangents, axis=1), 1.0)


def test_sinusoidal_curve_sampling_has_monotonic_arclength() -> None:
    curve = SinusoidalCurve(
        start_mm=(0.0, 0.0, 0.0),
        end_mm=(10.0, 0.0, 0.0),
        amplitude_mm=(0.0, 2.0, 0.0),
        periods=1.0,
    )
    centreline = curve.sample(step_mm=1.0)

    assert centreline.n_points >= 10
    assert np.all(np.diff(centreline.arclength_mm) >= 0.0)
    assert centreline.length_mm > 10.0


def test_centreline_to_dataframe_contains_expected_columns() -> None:
    curve = LineCurve(start_mm=(0.0, 0.0, 0.0), end_mm=(2.0, 0.0, 0.0))
    centreline = curve.sample(n_samples=3, object_id="target", segment_id="main")
    table = centreline.to_dataframe()

    assert {"object_id", "segment_id", "i_mm", "tangent_i"}.issubset(table.columns)
    assert table.shape[0] == 3


def test_cumulative_arclength_rejects_single_point() -> None:
    with pytest.raises(ValueError, match="At least two points"):
        cumulative_arclength_mm([[0.0, 0.0, 0.0]])


def test_centreline_rejects_non_monotonic_arclength() -> None:
    with pytest.raises(ValueError, match="monotonically increasing"):
        Centreline(
            coordinates_mm=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            tangents=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            arclength_mm=[0.0, -1.0],
            parameters=[0.0, 1.0],
        )
