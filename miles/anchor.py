"""Anchor management.

"""

__all__ = ['Anchor']

from typing import Optional

import numpy as np

from miles import CollectiveVariables


class Anchor:
    """Anchor point.

    An anchor is a point in the space of collective variables that
    serves as a seed for a Voronoi tessellation.

    Attributes
    ----------
    collective_variables : CollectiveVariables
        Space of collective variables where the anchor belongs.
    coordinates : np.array
        Coordinates of the anchor in the space of collective
        variables.
    index : int
        Unique index identifying the anchor.

    """

    def __init__(self, collective_variables: CollectiveVariables,
                 coordinates: np.array, index: Optional[int] = None) \
            -> None:
        self.collective_variables = collective_variables
        self.coordinates = coordinates
        self.index = index

    def __repr__(self) -> str:
        return ('{}({!r}, {!r}, index={!r})'
                .format(self.__class__.__name__,
                        self.collective_variables,
                        self.coordinates,
                        self.index))

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.index == other.index
        else:
            return False

    def __lt__(self, other) -> bool:
        return self.index < other.index

    def __hash__(self) -> int:
        return hash((tuple(self.coordinates), self.index))

    def distance(self, coordinates: np.array) -> float:
        """Distance from point to anchor.

        Parameters
        ----------
        coordinates : np.array
            Coordinates of a point in the space of collective
            variables.

        Returns
        -------
        d : float
            Distance from point to anchor.

        """
        return self.collective_variables.distance(coordinates,
                                                  self.coordinates)
