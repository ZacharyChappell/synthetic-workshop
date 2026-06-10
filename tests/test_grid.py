import numpy as np
import pytest

from synthworkshop.grid import GridSpec


def test_grid_defaults_for_3d() -> None:
    grid = GridSpec(shape=(4, 5, 6), spacing=(1.0, 2.0, 3.0))
    assert grid.ndim == 3
    assert grid.origin == (0.0, 0.0, 0.0)
    assert grid.axis_names == ("i", "j", "k")
    assert grid.n_voxels == 120


def test_grid_defaults_for_2d() -> None:
    grid = GridSpec(shape=(4, 5), spacing=(1.0, 2.0))
    assert grid.ndim == 2
    assert grid.axis_names == ("i", "j")


def test_grid_rejects_spacing_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="2 values"):
        GridSpec(shape=(4, 5), spacing=(1.0, 2.0, 3.0))


def test_index_and_world_arrays_have_expected_shape() -> None:
    grid = GridSpec(shape=(4, 5, 6), spacing=(1.0, 2.0, 3.0))
    index_arrays = grid.index_arrays()
    world_arrays = grid.world_arrays()

    assert len(index_arrays) == 3
    assert all(array.shape == grid.shape for array in index_arrays)
    assert np.allclose(world_arrays[1][:, 2, :], 4.0)


def test_index_to_world_and_world_to_index_round_trip() -> None:
    grid = GridSpec(
        shape=(10, 11, 12), spacing=(1.0, 2.0, 4.0), origin=(10.0, 20.0, 30.0)
    )
    index_coords = np.array([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
    world_coords = grid.index_to_world(index_coords)
    recovered = grid.world_to_index(world_coords)

    assert np.allclose(world_coords, [[10.0, 20.0, 30.0], [12.0, 26.0, 46.0]])
    assert np.allclose(recovered, index_coords)


def test_affine_matrix_contains_spacing_and_origin() -> None:
    grid = GridSpec(
        shape=(10, 11, 12), spacing=(1.0, 2.0, 4.0), origin=(10.0, 20.0, 30.0)
    )
    affine = grid.affine_matrix()

    assert affine.shape == (4, 4)
    assert np.allclose(np.diag(affine)[:3], [1.0, 2.0, 4.0])
    assert np.allclose(affine[:3, 3], [10.0, 20.0, 30.0])
