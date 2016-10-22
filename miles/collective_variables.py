"""Collective variables module.

This module deals with the space of collective variables. In
particular it keeps track of the different notions of distance for
each collective variable and computes the distance in the (Cartesian)
product metric space.

"""


__all__ = ['CollectiveVariables']

import math
from typing import Sequence

import numpy as np

from miles import CollectiveVariable, PhaseSpacePoint


class CollectiveVariables:
    """Space of collective variables.

    Attributes
    ----------
    dimension : int
        Dimensionality of the space of collective variables.
    collective_variables : Sequence[CollectiveVariable]
        Ordered collection of collective variables.
    indices : Sequence[int]
        Indices of the columns in the colvars.traj files where the
        collective variables appear. See note below.

    Notes
    -----
    The variable collective_variables[i] can be found at the
    indices[i]-th column of each .colvars.traj file.

    """
    def __init__(self, collective_variables: Sequence[CollectiveVariable],
                 indices: Sequence[int] = ()) -> None:
        """Initializes the space of collective variables.

        Parameters
        ----------
        collective_variables : Sequence[CollectiveVariable]
            Sequence of instances of the CollectiveVariable class.

        indices : Sequence[int]
            Indices of the columns in the output of colvars where the
            collective variables can be read off.

        """
        self.dimension = len(collective_variables)  # type: int
        self.collective_variables = collective_variables  # type: Sequence[CollectiveVariable]
        self.indices = indices
        if indices:
            assert len(indices) == self.dimension

    def __repr__(self) -> str:
        return ('{}({!r}, indices={!r})'
                .format(self.__class__.__name__,
                        self.collective_variables,
                        self.indices))

    def __call__(self, x: PhaseSpacePoint) -> np.array:
        """Map a phase space point into the space of collective variables.

        Parameters
        ----------
        x : PhaseSpacePoint
            A phase space point.

        Returns
        -------
        colvars : np.array
            Values of the collective variable at the point `x`.

        """
        return np.fromiter((cv(x) for cv in self.collective_variables),
                           dtype=np.float64)

    def __iter__(self):
        return iter(self.collective_variables)

    def __str__(self):
        s = '=' * 77 + '\n'

        fields = ['Name', 'Lower bound', 'Upper bound', 'Periodic?']
        s += '{:<14} | {:<12} | {:<12} | {}\n'.format(*fields)

        s += '-' * 77 + '\n'

        fmt = '{:<14} | {:<12} | {:<12} | {}\n'

        for cv in self.collective_variables:
            s += fmt.format(cv.name, cv.codomain.a, cv.codomain.b,
                            cv.codomain.periodic)

        s += '=' * 77

        return s

    def distance(self, a: np.array, b: np.array) -> float:
        """Distance between two points in the space of collective variables.

        Computes the $L_2$ product metric corresponding to the space
        of collective variables.

        Parameters
        ----------
        a : np.array
            Coordinates of a point in the space of collective
            variables.
        b : np.array
            Coordinates of a point in the space of collective
            variables.

        Returns
        -------
        d : float
            Distance (using the $L_2$ product metric) between the
            points `a` and `b`.

        """
        assert a.shape == b.shape

        d = 0.0

        for i, (ai, bi) in enumerate(zip(a, b)):
            di = self.collective_variables[i].codomain.distance(ai, bi)
            d += di * di

        return math.sqrt(d)
