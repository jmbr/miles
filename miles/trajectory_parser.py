"""Trajectory parser.

This module deals with the transformation of trajectory fragments into
sequences of Transition objects.

"""


__all__ = ['TrajectoryParser']

import logging
from typing import Generator, List, Optional, Sequence, Tuple  # noqa: F401

import numpy as np
from scipy.spatial import cKDTree

from miles import (Anchor, CollectiveVariables, Configuration, Milestones, Milestone, Transition, format_colvars)  # noqa: E501


def coroutine(func):
    """Decorator for coroutines."""
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        next(cr)
        return cr
    return start


class TrajectoryParser:
    """Parse transitions in trajectory fragments.

    Attributes
    ----------
    milestones : Milestones
        Milestone factory.
    configuration : Configuration
        Configuration options for the simulation.
    collective_variables : CollectiveVariables
        Space of collective variables.
    time_step_length : float
        Time step length (assumed to be constant during the
        simulation).
    kdtree : cKDTree
        K-D tree used to find nearest anchors fast.
    L : np.array
        Vector that must be added to the coordinates in the space of
        collective variable so that they lie in the cartesian product
        of $[0, b_i]$ where b_i is the upper bound in each direction $i$.
    parser : coroutine, optional
        Coroutine that parses sequence of anchors and yields
        transitions.
    _collect_crossing_events : bool
        Whether to collect crossing events (self-crossings) in
        addition to transitions (i.e., crossing events between
        neighboring milestones).

    """

    __slots__ = ('milestones', 'configuration',
                 'collective_variables', 'time_step_length', 'kdtree',
                 'L', 'parser', '_collect_crossing_events')

    def __init__(self, milestones: Milestones,
                 configuration: Configuration,
                 collective_variables: CollectiveVariables) -> None:
        self.milestones = milestones
        self.configuration = configuration
        self.collective_variables = collective_variables

        self.time_step_length = configuration.time_step_length

        self.kdtree, self.L = make_kdtree(self.milestones.anchors)

        self.parser = None      # type: Optional[Generator]
        self._collect_crossing_events = False

    def __repr__(self) -> str:
        return ('{}({!r}, {!r}, {!r})'
                .format(self.__class__.__name__,
                        self.milestones,
                        self.configuration,
                        self.collective_variables))

    def setup(self, colvars: Optional[np.array] = None,
              milestone: Optional[Milestone] = None) -> None:
        # def setup(self, initial_transition: Optional[Transition]) -> None:
        """Initialize parser.

        """
        if colvars is not None and milestone is not None:
            _, idx = self.kdtree.query(colvars + self.L)  # , n_jobs=-1)
            initial_anchor = idx
            initial_milestone = tuple(a.index for a in milestone.anchors)
            self.parser = self.anchors(initial_anchor, initial_milestone)
        else:
            # We don't know the initial anchor and the initial
            # milestone, so we initialize the parser with dummy
            # values.
            self.parser = self.anchors(-1, (-1, -1))

    @property
    def collect_crossing_events(self) -> bool:
        return self._collect_crossing_events

    @collect_crossing_events.setter
    def collect_crossing_events(self, value: bool):
        self._collect_crossing_events = value

    def parse(self, coordinates: np.array) -> Sequence[Transition]:
        """Parse trajectory and extract transitions.

        Parameters
        ----------
        coordinates : np.array
            Array of coordinates of the points in the space of
            collective variables.

        Returns
        -------
        transitions : List[Transition]
            List of transitions that were found.

        """
        transitions = []        # type: List[Transition]

        # Obtain array containing indices of the closest anchors for
        # each frame, then compute the frame indices at which
        # transitions occur.
        distances, closest_anchor_indices \
            = self.kdtree.query(coordinates + self.L)  # , n_jobs=-1)

        for index, (anchor_index, distance, colvars) in \
                enumerate(zip(closest_anchor_indices, distances, coordinates)):
            data = Data(index, anchor_index, colvars, distance, transitions)
            self.parser.send(data)

        return transitions

    @coroutine
    def anchors(self, initial_anchor: int,
                initial_milestone: Tuple[int, int]) -> Generator:
        """Transform anchors into pairs.

        """
        ps = self.pairs(initial_anchor, initial_milestone)

        while True:
            ps.send((yield))

    @coroutine
    def pairs(self, prev_anchor: int,
              initial_milestone: Tuple[int, int]) -> Generator:
        """Emit anchor pairs.

        """
        def make_milestone(a, b):
            return tuple(sorted((a, b)))

        cs = self.crossings(initial_milestone)
        step = 1

        while True:
            data = (yield)
            data.milestone = make_milestone(prev_anchor, data.anchor)
            data.step = step
            prev_anchor = data.anchor
            # print('pair', data.step, data.milestone,
            #       'step', step, 'prev_anchor', prev_anchor,
            #       'colvars', format_colvars(data.colvars))
            cs.send(data)
            step += 1

    @coroutine
    def crossings(self, initial_milestone: Tuple[int, int]) -> Generator:
        """Emit crossings.

        """
        def make_crossing(f, point):
            final_milestone = self.milestones.make_from_indices(f[0], f[1])
            return Transition(final_milestone, None, point, None)

        ts = self.transitions(initial_milestone)

        while True:
            data = (yield)
            a, b = data.milestone
            if a != b and valid_indices(data.milestone):
                # print('crossing', data.step, data.milestone)
                if self.collect_crossing_events:
                    crossing = make_crossing((a, b), data.colvars)
                    data.results.append((data.index, crossing))
                ts.send(data)

    @coroutine
    def transitions(self, prev_milestone: Tuple[int, int]) -> Generator:
        """Emit transitions.

        """
        def valid_transition(indices0, indices1):
            return valid_indices(indices0) and valid_indices(indices1)

        def make_transition(f, i, point, lag_steps):
            b = self.milestones.make_from_indices(f[0], f[1])
            a = self.milestones.make_from_indices(i[0], i[1])
            t = lag_steps * self.time_step_length
            return Transition(b, a, point, t)

        prev_step = 0
        # print('transitions step', None, 'milestone', None,
        #       'prev_step', prev_step, 'prev_milestone',
        #       prev_milestone, 0, '<==')

        while True:
            data = (yield)
            if (data.milestone != prev_milestone
                    and valid_transition(data.milestone, prev_milestone)):
                # print('transitions step', data.step, 'milestone',
                #       data.milestone, 'prev_step', prev_step,
                #       'prev_milestone', prev_milestone,
                #       data.step - prev_step, '<--')
                t = make_transition(data.milestone, prev_milestone,
                                    data.colvars, data.step - prev_step)
                data.results.append((data.index, t))
                prev_step, prev_milestone = data.step, data.milestone


class Data:
    """Data used internally by the trajectory parser.

    """
    __slots__ = ('index', 'anchor', 'colvars', 'distance', 'results',
                 'milestone', 'step')

    def __init__(self, index: int, anchor: int, colvars: np.array,
                 distance: float, results: List[Transition],
                 milestone: Optional[Tuple[int, int]] = None,
                 step: Optional[int] = None) -> None:
        self.index = index
        self.anchor = anchor
        self.colvars = colvars
        self.distance = distance
        self.results = results
        self.milestone = milestone
        self.step = step

    def __repr__(self) -> str:
        return ('{}({!r}, {!r}, {!r}, {!r}, {!r}, '
                'milestone_indices={!r}, step={!r})'
                .format(self.__class__.__name__,
                        self.index,
                        self.anchor,
                        self.colvars,
                        self.distance,
                        self.results,
                        self.milestone,
                        self.step))


def valid_indices(indices: Tuple[int, int]) -> bool:
    """Returns true if the indices corresponding to a milestone are valid

    """
    return -1 not in indices


def compute_bounds(collective_variables: CollectiveVariables,
                   non_periodic_value=-1) -> Tuple[np.array, np.array]:
    """Compute bounds for each collective variable.

    Determine which collective variables are periodic and compute
    bounds so that the periodic KD tree knows about it.

    Parameters
    ----------
    collective_variables : CollectiveVariables
        Space of collective variables.
    non_periodic_value : float
        Value to be used to signal non-periodicity in a direction.

    Returns
    -------
    bounds : np.array
        Bounds suitable for use by PeriodicKDTree

    L : np.array
        Vector to add to the data so that it lies within the cartesian
        product of [0, bounds[i]] for all i.

    """
    dim = collective_variables.dimension
    bounds = np.zeros(dim)
    L = np.zeros(dim)

    for i, cv in enumerate(collective_variables):
        if cv.codomain.periodic:
            bounds[i] = cv.codomain.b - cv.codomain.a
            L[i] = -cv.codomain.a
        else:
            bounds[i] = non_periodic_value
            L[i] = 0.0

    return bounds, L


def make_kdtree(anchors: Sequence[Anchor]) -> Tuple[cKDTree, np.array]:
    """Create a KDTree.

    This function instantiates a cKDTree taking into account periodic
    boundary conditions and working around certain bugs in older
    implementations of scipy.spatial.cKDTree.

    Parameters
    ----------
    anchors : Sequence[Anchor]
        Anchor points to be used in the nearest neighbor search.

    Returns
    -------
    kdtree : cKDTree
        An initialized instance of cKDTree.

    L : np.array
        Vector to add to the data so that it lies within the cartesian
        product of [0, bounds[i]] for all i.

    """
    collective_variables = anchors[0].collective_variables

    anchor_colvars = np.array([anchor.coordinates for anchor in anchors])

    try:
        bounds, L = compute_bounds(collective_variables)
        kdtree = cKDTree(anchor_colvars + L, boxsize=bounds)
    except ValueError:
        bounds, L = compute_bounds(collective_variables,
                                   non_periodic_value=np.inf)
        kdtree = cKDTree(anchor_colvars + L,
                         boxsize=np.concatenate((bounds, bounds)))

    return kdtree, L
