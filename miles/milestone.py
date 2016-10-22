"""Milestone module

"""


__all__ = ['Milestone']

from typing import Callable     # noqa: F401

import numpy as np

from miles import Anchor


class Milestone:
    """Milestone.

    A milestone is uniquely identified by a pair of anchors.
    Associated to these anchors there is a unique integer number (the
    index).

    Always instantiate :class:`Milestone` via the :class:`Milestones`
    factory instead of doing so directly.

    Attributes
    ----------
    anchors : Sequence[Anchor]
        A sequence of two anchors sorted by their index.
    index : int
        Unique index identifying the milestone.

    """

    def __init__(self, anchor0: Anchor, anchor1: Anchor, index: int) -> None:
        self.anchors = sorted([anchor0, anchor1],
                              key=lambda a: a.index)
        self.index = index
        self._projector = None  # type: Optional[Callable[[np.array], np.array]] # noqa: E501

    def __repr__(self) -> str:
        assert len(self.anchors) == 2
        a, b = self.anchors
        return '{}({!r}, {!r}, {!r})'.format(self.__class__.__name__,
                                             a, b, self.index)

    def __str__(self) -> str:
        i = [anchor.index for anchor in self.anchors]
        # return 'Milestone {} ({}, {})'.format(self.index, i[0], i[1])
        return 'Milestone ({}, {})'.format(i[0], i[1])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Milestone):
            raise NotImplementedError

        A = frozenset(a.index for a in self.anchors)
        B = frozenset(b.index for b in other.anchors)

        return A == B

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Milestone):
            raise NotImplementedError

        return self.index < other.index

    def __hash__(self) -> int:
        return hash(self.index)

    def _make_projector(self) -> Callable[[np.array], np.array]:
        """Projection operator onto milestone.

        Returns
        -------
        projector : Callable[[np.array], np.array]
            Projection operator onto the hyperplane orthogonal to the
            vector joining the anchors that define the given milestone.

        """
        a = self.anchors[0].coordinates
        b = self.anchors[1].coordinates

        v = b - a
        v /= np.linalg.norm(v)

        center = (a + b) / 2.0

        def projector(x: np.array) -> np.array:
            return x - v * v.dot(x) + center

        return projector

    def project(self, point: np.array) -> np.array:
        """Project a point onto the milestone.

        Project a point in the space of collective variables onto the
        milestone.

        Parameters
        ----------
        point : np.array
            Point to be projected.

        Returns
        -------
        projection : np.array
            Projected point on the milestone.

        """
        if not self._projector:
            self._projector = self._make_projector()

        return self._projector(point)
