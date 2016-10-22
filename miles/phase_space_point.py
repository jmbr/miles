"""Phase space point.

"""


__all__ = ['PhaseSpacePoint']

from typing import Optional

import numpy as np

from miles import Mapping


class PhaseSpacePoint:
    """A phase space point.

    """
    __slots__ = ('positions', 'velocities', '_mapping', '_colvars')

    def __init__(self, positions: np.array,
                 velocities: np.array,
                 mapping: Mapping,
                 colvars: Optional[np.array] = None) -> None:
        self.positions = positions
        self.velocities = velocities
        self._mapping = mapping
        self._colvars = colvars

    def __repr__(self) -> str:
        return ('{}({!r}, {!r}, {!r}, colvars={!r})'
                .format(self.__class__.__name__,
                        self.positions,
                        self.velocities,
                        self._mapping,
                        self._colvars))

    @property
    def colvars(self) -> np.array:
        if self._colvars is None:
            self._colvars = self._mapping(self)
        return self._colvars
