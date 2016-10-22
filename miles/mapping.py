__all__ = ['MappingDihedrals', 'Mapping', 'MappingPChi', 'MappingX', 'MappingXY']  # noqa: E501

import logging
from typing import Optional     # noqa: F401
from abc import ABCMeta, abstractmethod

import numpy as np

import miles.dihedrals as dihedrals


class Mapping(metaclass=ABCMeta):
    colvars_dim = None          # type: Optional[int]

    @abstractmethod
    def __call__(self, phase_space_point):
        raise NotImplementedError


class MappingPChi(Mapping):
    """Map onto pseudo rotation angle and glycosil torsion."""
    colvars_dim = 2

    def __init__(self):
        logging.info('Using (p, chi)-coordinates')

    def __call__(self, phase_space_point):
        coords = phase_space_point.positions
        p = dihedrals.compute_pseudo_rotation(coords)
        chi = dihedrals.compute_chi(coords)

        return np.array([p, chi])

    # def __call__(self, coords):
    #     return dihedrals.compute_chi_and_pseudo_rotation(coords)


class MappingX(Mapping):
    """Map onto x coordinate."""
    colvars_dim = 2

    def __call__(self, phase_space_point):
        return phase_space_point.positions[0:1]


class MappingXY(Mapping):
    """Map onto x, y coordinates."""
    def __init__(self):
        logging.info('Using (x, y)-coordinates')

    def __call__(self, phase_space_point):
        return phase_space_point.positions[0:2]


class MappingDihedrals(Mapping):
    """Map onto user-specified dihedral angles."""
    def __init__(self, indices):
        self.indices = indices
        self.colvars_dim = len(indices)
        logging.info('Using dihedrals: {}'.format(indices))

    def __call__(self, phase_space_point):
        pos = phase_space_point.positions

        dih = dihedrals.compute_dihedrals(pos, self.indices)

        return dih
