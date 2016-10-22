"""Transition kernel class.

A transition kernel allows us to estimate the (full-resolution)
transition operator in milestoning.

The main goal of this class is to keep track of the transitions and
obtain:
 1. the stationary flux vector,
 2. an approximation of the stationary distribution, and
 3. the estimated mean first passage time.

"""


__all__ = ['TransitionKernel', 'TransitionKernelError']

import logging
import os
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple, Union  # noqa: F401

import numpy as np
import scipy.sparse.linalg

from miles import (Database, Distribution, Distributions, Matrices, Milestone, Transition, save_distributions)  # noqa: E501


class TransitionKernel(defaultdict,
                       Dict[Milestone, Dict[Milestone, Distribution]]):
    """Transition kernel.

    This class is a container for transitions between milestones.
    This is a generalization of the idea of a transition matrix. In
    our case, our kernels map a distribution to a pair of final and
    initial milestones. That is, the following is true::

    isinstance(TransitionKernel[Milestone][Milestone], Distribution)

    Attributes
    ----------
    final_milestones : Set
        The set of milestones determining the product state.
    initial_milestones : Set
        The set of milestones comprising the reactant state.
    matrices : Matrices
        Required matrices for keeping track of transitions between
        milestones and local MFPTs.

    Examples
    --------
    >>> transition_kernel = TransitionKernel(matrices)
    >>> transition_kernel[final_milestone][initial_milestone].sample()

    will return a random sample from the distribution of first hitting
    points on final_milestone that originate from initial_milestone.

    """
    def __init__(self,
                 matrices_or_database: Union[Matrices, Database]) -> None:
        """Instantiate a new TransitionKernel object.

        Parameters
        ----------
        matrices_or_database
            This might be an instance of the Matrices class or the
            Database class. If it is the latter, then the transition
            kernel is initialized using the entries found in the the
            database.

        """
        super(TransitionKernel, self).__init__(lambda:
                                               defaultdict(Distribution))

        # Known final and initial milestones.
        self.final_milestones = set()  # type: Set[Milestone]
        self.initial_milestones = set()  # type: Set[Milestone]

        if isinstance(matrices_or_database, Matrices):
            self.matrices = matrices_or_database
        elif isinstance(matrices_or_database, Database):
            self._from_database(matrices_or_database)
        else:
            raise TypeError('Unable to initialize {} from {}'
                            .format(self.__class__.__name__,
                                    matrices_or_database))

    def _from_database(self, database: Database) -> None:
        """Populate the transition kernel using a database."""
        self.matrices = Matrices(database.milestones)

        # logging.debug('Populating transition sampler with entries
        # from {}'.format(database))

        entries = database.get_entries()
        for final_milestone in entries.keys():
            for transition in entries[final_milestone]:
                self.update(transition)

    def update(self, transition: Transition) -> None:
        """Add a new transition to the kernel.

        This method simultaneously updates the finite-dimensional
        transition matrix as well as the empirical estimate of the
        transition kernel.

        Parameters
        ----------
        transition
            A particular transition between two milestones.

        """
        f = transition.final_milestone
        i = transition.initial_milestone

        self.final_milestones.add(f)
        self.initial_milestones.add(i)

        # # Since the transition is assumed to be stored in the
        # # database, we don't need to keep the auxiliary files around
        # # in the file system.
        # assert transition.file_names
        # transition.remove_files()

        self[f][i].update(transition)

        self.matrices.update(transition)

    def compute_distributions(self, prev_q: Optional[np.array] = None) \
            -> Tuple[Distributions, np.array]:
        """Returns estimated stationary distribution.

        Parameters
        ----------
        prev_q : np.array, optional
            Vector of weights of the distributions on the milestones
            at the previous iteration.

        Returns
        -------
        distributions : Distributions
            Approximation to the stationary distributions on each
            milestone.
        q : np.array
            Vector of weights of the distributions on the milestones
            to be used at the next iteration.

        Raises
        ------
        TransitionKernelError
            If it is not possible to compute the dominant eigenvector
            for the transition matrix corresponding to the current
            transition kernel.

        """
        logging.debug('Computing dominant eigenvector.')

        if prev_q is not None:
            self.matrices.compute(prev_q)
        else:
            try:
                self.matrices.compute()
            except (ValueError,
                    scipy.sparse.linalg.ArpackNoConvergence) as exc:
                logging.warning('Could not solve finite-dimensional '
                                'eigenvalue problem: {}.'.format(exc))
                raise TransitionKernelError('ARPACK did not converge') from exc

        q = self.matrices.q

        distributions = Distributions()

        for j in self.final_milestones:
            distribution = Distribution()

            for i in self[j].keys():
                assert i != j, ('Invalid transition kernel: a trajectory '
                                'starts and stops at the same milestone.')

                ii, jj = i.index, j.index
                Kij = self.matrices.K[ii, jj]

                wi = q[ii]
                weight = wi * Kij

                Fij = self[j][i]
                distribution += weight * Fij

            distributions[j] = distribution

        return distributions, q

    def distribution(self, initial_milestone: Milestone) -> Distribution:
        """Distribution of first hitting points obtained by sampling from a
        specified milestone.

        """
        K = self.matrices.K

        distribution = Distribution()

        i = initial_milestone

        for j in self.final_milestones:
            if i not in self[j]:
                continue

            # XXX The order of the indices in the finite dimensional
            # matrix is the opposite to those of the transition
            # operator.
            ii, jj = i.index, j.index
            distribution += K[ii, jj] * self[j][i]

        return distribution

    def distributions(self):
        """Returns an iterator over the distributions in the kernel."""
        for f in self.keys():
            for i in self[f].keys():
                yield self[f][i]

    def save_distributions(self, suffix=''):
        """Save first hitting point / lag time distributions to files."""
        for fin in self.keys():
            for ini in self[fin].keys():
                ds = Distributions()
                ds[fin] = self[fin][ini]

                stem = ('kernel-{}-{}{}'
                        .format(ini.index, fin.index, suffix))
                fname = os.path.extsep.join([stem, 'dst'])
                save_distributions(ds, fname)


class TransitionKernelError(Exception):
    pass
