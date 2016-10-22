"""Compute different kinds of dihedral angles.

"""


__all__ = ['compute_chi', 'compute_pseudo_rotation', 'compute_dihedral', 'compute_dihedrals']  # noqa: E501

import numpy as np


SIN_ALPHA1 = np.sin(np.deg2rad(36.0))
SIN_ALPHA2 = np.sin(np.deg2rad(72.0))


def compute_dihedral(xyz, indices):
    """Compute a dihedral angle."""
    vv = np.squeeze(xyz[indices, :])
    b = np.diff(vv, axis=0)

    b0_cross_b1 = np.cross(b[0, :], b[1, :])
    b1_cross_b2 = np.cross(b[1, :], b[2, :])

    b1unit = b[1, :] / np.linalg.norm(b[1, :], 2)

    term1 = np.dot(np.cross(b0_cross_b1, b1_cross_b2), b1unit)
    term2 = np.dot(b0_cross_b1, b1_cross_b2)

    return np.rad2deg(np.arctan2(term1, term2))


def compute_dihedrals(xyz, all_indices):
    """Compute many dihedral angles."""
    dihedrals = np.zeros(len(all_indices), dtype=np.float64)

    for i, indices in enumerate(all_indices):
        dihedrals[i] = compute_dihedral(xyz, indices)

    return dihedrals


def compute_chi(xyz):
    """Compute glycosil dihedral.

    Computes the dihedral angle involving the atoms O4'-C1'-N9-C4

    """
    # indices = [[7], [8], [10], [23]]
    indices = [[7], [8], [10], [19]]
    return np.rad2deg(compute_dihedral(xyz, indices))


def compute_pseudo_rotation(xyz):
    """Compute pseudo-rotation.

    Calculate pseudo rotation involving the atoms C1'-C2'-C3'-C4'-O4'.

    """
    A = [[8], [24], [28], [5], [7]]  # C1', C2', C3', C4', O4'

    v1 = compute_dihedral(xyz, [A[0], A[1], A[2], A[3]])  # C1'-C2'-C3'-C4'
    v2 = compute_dihedral(xyz, [A[1], A[2], A[3], A[4]])  # C2'-C3'-C4'-O4'
    v3 = compute_dihedral(xyz, [A[2], A[3], A[4], A[0]])  # C3'-C4'-O4'-C1'
    v4 = compute_dihedral(xyz, [A[3], A[4], A[0], A[1]])  # C4'-O4'-C1'-C2'
    v5 = compute_dihedral(xyz, [A[4], A[0], A[1], A[2]])  # O4'-C1'-C2'-C3'

    return np.rad2deg(np.arctan2(-v2 + v3 - v4 + v5,
                                 2.0 * v1 * (SIN_ALPHA1 + SIN_ALPHA2)))
