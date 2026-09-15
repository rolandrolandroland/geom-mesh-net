"""Regression tests for density_grid.generate_density_grid (formerly voxelize_clusters).

The function is deprecated as a ground truth (``experiments/reconstruction/ROADMAP.md``
section 3.2) but still used by ``LoadData``. Two properties are pinned: it no
longer raises on the 53 to 60 stored patterns containing a zero or tiny radius,
and it says, when called, that it is not the simulator's field.
"""

import numpy as np
import pytest

from geom_mesh_net.fields import density_grid as vc


def grid(centres, radii):
    return vc.generate_density_grid(grid_size=[10, 10, 10], cluster_centers=centres, radii=np.asarray(radii),
                                    rho_c=0.5, rho_b=0.02, resolution=1)[3]


def test_zero_radius_cluster_no_longer_raises_and_changes_nothing():
    one = {"x": np.array([5.0]), "y": np.array([5.0]), "z": np.array([5.0])}
    with_empty = {"x": np.array([5.0, 2.3]), "y": np.array([5.0, 2.3]), "z": np.array([5.0, 2.3])}
    with pytest.warns(DeprecationWarning):
        expected = grid(one, [3.0])
    with pytest.warns(DeprecationWarning):
        result = grid(with_empty, [3.0, 0.0])  # no voxel centre within 0 of 2.3
    assert np.array_equal(result, expected)


def test_warns_that_it_is_not_the_simulators_field():
    centres = {"x": np.array([5.0]), "y": np.array([5.0]), "z": np.array([5.0])}
    with pytest.warns(DeprecationWarning, match="fields.oracle.replay_oracle"):
        grid(centres, [3.0])
