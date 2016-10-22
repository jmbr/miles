"""Milestones module

This source file contains the code to set up and instantiate a
collection of milestones.

"""


__all__ = ['Milestones', 'MilestonesError']

import ast
import logging
from typing import Optional, List, Sequence

from miles import (Anchor, Milestone)


class MilestonesError(Exception):
    pass


class Milestones:
    """Factory of milestones.

    This class is basically a container for milestones, where each
    milestone is uniquely defined by a pair of anchors.

    The class contains references to milestones acting as reactants
    and products.

    Attributes
    ----------
    anchors : List[Anchor]
        List of all anchors.
    reactants : List[Milestone]
        List of all milestones comprising the reactant state.
    products : List[Milestone]
        List of milestones comprising the product state.
    _counter : int
        Keeps track of how many anchors are known.

    """
    __slots__ = ('anchors', 'reactants', 'products', '_counter')

    def __init__(self, anchors: Optional[List[Anchor]] = None,
                 reactants=None, products=None) -> None:
        if anchors:
            self.anchors = anchors
        else:
            self.anchors = []

        if reactants:
            self.reactants = reactants
        else:
            self.reactants = []

        if products:
            self.products = products
        else:
            self.products = []

        self._counter = 0       # WARNING: Not reentrant.

    def __repr__(self) -> str:
        return ('{}(anchors={!r}, reactants={!r}, products={!r})'
                .format(self.__class__.__name__, self.anchors,
                        self.reactants, self.products))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Milestones):
            raise NotImplementedError

        return (self.anchors == other.anchors and
                self.reactants == other.reactants and
                self.products == other.products and
                self._counter == other._counter)

    def append_anchor(self, anchor: Anchor) -> None:
        """Append an anchor.

        Each anchor is identified by a unique number called index. A
        call to this method will make the factory of milestones aware
        of the anchor and will alter its index.

        Parameters
        ----------
        anchor : Anchor
            The anchor to append to the collection. The anchor's index
            attribute will be altered by a call to this method.

        """
        if anchor.index is None:
            anchor.index = self._counter
            self._counter += 1
        self.anchors.append(anchor)

    def append_anchors(self, anchors: Sequence[Anchor]) -> None:
        """Append many anchors to the collection."""
        for anchor in anchors:
            self.append_anchor(anchor)

    def _get_index(self, idx0: int, idx1: int) -> int:
        """Compute unique milestone index from two anchor indices.

        """
        num_anchors = len(self.anchors)
        lst = sorted([idx0, idx1])
        assert idx0 < num_anchors and idx1 < num_anchors
        idx = lst[1] + num_anchors * lst[0]

        return idx

    def make(self, a: Anchor, b: Anchor) -> Milestone:
        """Instantiate a milestone using two anchors.

        Parameters
        ----------
        a : Anchor
            One of the required anchors.
        b : Anchor
            Another anchor.

        Returns
        -------
        milestone : Milestone
            A newly instantiated milestone.

        Raises
        ------
        MilestonesError
            In case one of the anchors is invalid.

        """
        for anchor in [a, b]:
            if anchor not in self.anchors:
                raise MilestonesError('Unknown anchor:', anchor)

        idx0 = self.anchors.index(a)
        idx1 = self.anchors.index(b)
        idx = self._get_index(idx0, idx1)

        return Milestone(a, b, idx)

    def make_from_indices(self, idx0: int, idx1: int) -> Milestone:
        """Instantiate a milestone from two anchor indices."""
        a = self.anchors[idx0]
        b = self.anchors[idx1]
        idx = self._get_index(idx0, idx1)

        return Milestone(a, b, idx)

    def make_from_index(self, idx: int) -> Milestone:
        """Instantiate a milestone from a single index.

        The index must be of the form A + B * max_milestone, where A
        and B are anchor indices.

        """
        t = divmod(idx, len(self.anchors))
        return self.make_from_indices(t[0], t[1])

    def make_from_str(self, arg: str) -> Milestone:
        """Instantiate a milestone from a string.

        Parameters
        ----------
        arg : str
            String representing the milestone. This can be either a pair
            of integer indices of anchors (e.g., "7, 8") or a single
            integer representing the milestone (e.g., "71").

        Returns
        -------
        milestone : Milestone
            A Milestone object corresponding to the specified
            milestone.

        Raises
        ------
        MilestonesError
            If the requested milestone is not valid.

        """
        m = ast.literal_eval(arg)

        if isinstance(m, tuple):
            milestone = self.make_from_indices(m[0], m[1])
        elif isinstance(m, int):
            milestone = self.make_from_index(m)
        else:
            raise MilestonesError('Invalid milestone {!r}'.format(m))

        return milestone

    @property
    def max_milestones(self) -> int:
        """Return the maximum possible number of milestones.

        This is used to establish the dimensions of the transition
        matrices and the associated vectors.

        """
        # Note that the actual upper bound on the number of milestones
        # could also be n * (n - 1) / 2 at the cost of more involved
        # indexing.
        return len(self.anchors)**2

    def append_reactant(self, milestone: Milestone) -> None:
        """Add one milestone to the reactants."""
        logging.debug('Appending {} to the list of reactant milestones.'
                      .format(milestone))
        self.reactants.append(milestone)

    def append_product(self, milestone: Milestone) -> None:
        """Add one milestone to the products."""
        logging.debug('Appending {} to the list of product milestones.'
                      .format(milestone))
        self.products.append(milestone)
