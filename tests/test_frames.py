from __future__ import annotations

import numpy as np
import pytest

from synthworkshop.primitives.curves import LineCurve
from synthworkshop.primitives.frames import build_reference_guided_frame


def test_reference_guided_frame_for_i_axis_line() -> None:
    centreline = LineCurve(
        start_mm=(0.0, 0.0, 0.0),
        end_mm=(10.0, 0.0, 0.0),
    ).sample(n_samples=5)

    frame = build_reference_guided_frame(centreline)

    assert frame.n_points == centreline.n_points
    assert np.all(frame.valid)
    assert np.allclose(frame.tangents[:, 0], 1.0)
    assert np.allclose(frame.primary_axes[:, 1], 1.0)
    assert np.allclose(frame.secondary_axes[:, 2], 1.0)


def test_reference_guided_frame_axes_are_orthonormal() -> None:
    centreline = LineCurve(
        start_mm=(0.0, 0.0, 0.0),
        end_mm=(2.0, 3.0, 4.0),
    ).sample(n_samples=7)

    frame = build_reference_guided_frame(centreline)

    assert np.allclose(np.linalg.norm(frame.tangents, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(frame.primary_axes, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(frame.secondary_axes, axis=1), 1.0)
    assert np.allclose(np.sum(frame.tangents * frame.primary_axes, axis=1), 0.0)
    assert np.allclose(np.sum(frame.tangents * frame.secondary_axes, axis=1), 0.0)
    assert np.allclose(np.sum(frame.primary_axes * frame.secondary_axes, axis=1), 0.0)


def test_reference_guided_frame_uses_fallback_axis_when_parallel() -> None:
    centreline = LineCurve(
        start_mm=(0.0, 0.0, 0.0),
        end_mm=(0.0, 10.0, 0.0),
    ).sample(n_samples=5)

    frame = build_reference_guided_frame(
        centreline,
        reference_axes=((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )

    assert np.allclose(frame.primary_axes[:, 2], 1.0)


def test_reference_guided_frame_rejects_2d_centreline() -> None:
    centreline = LineCurve(start_mm=(0.0, 0.0), end_mm=(1.0, 0.0)).sample(n_samples=3)

    with pytest.raises(ValueError, match="3D curves"):
        build_reference_guided_frame(centreline)


def test_frame_to_dataframe_contains_expected_columns() -> None:
    centreline = LineCurve(
        start_mm=(0.0, 0.0, 0.0),
        end_mm=(2.0, 0.0, 0.0),
    ).sample(n_samples=3)
    frame = build_reference_guided_frame(centreline)
    table = frame.to_dataframe()

    assert {"tangent_i", "primary_j", "secondary_k", "valid"}.issubset(table.columns)
    assert table.shape[0] == 3
