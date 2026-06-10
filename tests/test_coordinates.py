import numpy as np
import pytest

from synthworkshop.coordinates import (
    as_coordinate_array,
    normalise_vectors,
    validate_axis_names,
    validate_origin,
    validate_shape,
    validate_spacing,
)


def test_validate_shape_accepts_2d_and_3d() -> None:
    assert validate_shape([4, 5]) == (4, 5)
    assert validate_shape([4, 5, 6]) == (4, 5, 6)


def test_validate_shape_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        validate_shape([4, 0, 6])


def test_validate_spacing_requires_positive_finite_values() -> None:
    assert validate_spacing([1, 2, 3]) == (1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="finite and positive"):
        validate_spacing([1.0, np.inf, 1.0])


def test_validate_origin_requires_matching_dimension() -> None:
    assert validate_origin([0, 1, 2], ndim=3) == (0.0, 1.0, 2.0)
    with pytest.raises(ValueError, match="3 values"):
        validate_origin([0, 1], ndim=3)


def test_validate_axis_names_requires_unique_names() -> None:
    assert validate_axis_names(["i", "j", "k"], ndim=3) == ("i", "j", "k")
    with pytest.raises(ValueError, match="unique"):
        validate_axis_names(["i", "i", "k"], ndim=3)


def test_as_coordinate_array_accepts_single_coordinate() -> None:
    coords = as_coordinate_array([1, 2, 3], ndim=3)
    assert coords.shape == (3,)


def test_as_coordinate_array_accepts_coordinate_table() -> None:
    coords = as_coordinate_array([[1, 2, 3], [4, 5, 6]], ndim=3)
    assert coords.shape == (2, 3)


def test_normalise_vectors_returns_unit_vectors() -> None:
    vectors = normalise_vectors([[3, 4, 0], [0, 0, 2]])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_normalise_vectors_rejects_zero_vectors() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        normalise_vectors([[0, 0, 0]])
