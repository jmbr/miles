"""Transition sampler module.

This module contains the code necessary for sampling trajectory
fragments joining initial points to first hitting points.

A TransitionSampler object knows how to obtain an exact number of
valid trajectory fragments.

"""


__all__ = ['TransitionSampler']

import logging
from typing import Sequence, Optional  # noqa: F401. Used for mypy.

import numpy as np
from scipy.linalg import norm

from miles import (Clock, DatabaseAlreadyExistsError, Distribution, Milestone, Simulation, TimestepperServer, Transition, TransitionKernel, make_stochastic_matrix, make_time_matrix, vector_to_list)  # noqa: E501


class TransitionSampler:
    """Transition sampler."""

    def __init__(self, simulation: Simulation,
                 timestepper_server: TimestepperServer) -> None:
        self.simulation = simulation

        self.database = self.simulation.database
        self.output_dir = self.simulation.configuration.simulation_dir
        self.reactant_distribution = simulation.reactant_distribution
        self.products = simulation.milestones.products
        self.timestepper_server = timestepper_server

        # The following two variables are used to determine local
        # convergence.
        self._prev_Krow = None  # type: Optional[np.array]
        self._prev_Trow = None  # type: Optional[np.array]

    def __repr__(self) -> str:
        return ('{}({!r})' .format(self.__class__.__name__,
                                   self.simulation))

    def sample(self, min_samples: int,
               milestone: Milestone,
               distribution: Distribution,
               kernels: Sequence[TransitionKernel],
               tolerance: float) -> Sequence[TransitionKernel]:
        """Draw a number of samples (transitions).

        This method repeatedly samples transitions until the total
        number of valid samples drawn is at least equal to
        `min_samples` and the convergence criterion (determined by
        `tolerance`) is satisfied. All the transitions obtained are
        used to populate the transition kernels passed within the
        `kernels` list.

        Parameters
        ----------
        min_samples : int
            Minimum number of samples to draw.
        milestone :  Milestone
            Initial milestone from which to start trajectory
            fragments.
        distribution : Distribution
            Distribution of first hitting points at the milestone.
            This is the distribution from which we will be drawing the
            initial points of our trajectory fragments.
        kernels : Sequence[TransitionKernel]
            Instances of TransitionKernel (one for each parallel
            process).
        tolerance : float
            Threshold to use in the convergence criterion.

        Returns
        -------
        kernels : Sequence[TransitionKernel]
            List of instances of TransitionKernel updated according to
            the trajectory fragments observed.

        """
        assert distribution
        assert milestone == distribution.sample().final_milestone

        server = self.timestepper_server

        def free(sent, received):
            """Number of idle servers."""
            return server.num_servers - (sent - received)

        def has_converged() -> bool:
            return self._has_converged(kernels, milestone,
                                       tolerance, min_samples)

        def sanity_check(transitions, milestone) -> None:
            if not transitions:
                return

            assert transitions[0].initial_milestone == milestone, \
                ('Inconsistency in transitions: "{}" does not start from {}'.
                 format(transitions[0], milestone))

            for i in range(1, len(transitions)):
                prev = transitions[i-1]
                curr = transitions[i]
                assert prev.final_milestone == curr.initial_milestone, \
                    'Invalid chain of transitions.'

        def log_progress(sent, received, remaining):
            logging.debug('available: {:3d}\tsent: {:3d}\t'
                          'received: {:3d}\tremaining: {:3d}'
                          .format(server.num_servers - (sent - received),
                                  sent, received, remaining))

        has_converged()         # Initialize. XXX

        remaining = min_samples
        sent, received = 0, 0

        clock = Clock()
        clock.tic()

        while True:
            log_progress(sent, received, remaining)

            should_block = free(sent, received) == 0 or remaining == 0
            result = server.get(block=should_block)

            if result:
                received += 1

                _, transitions = result

                sanity_check(transitions, milestone)

                for transition in transitions:
                    if transition.initial_milestone in self.products:
                        break

                    if transition.is_valid():
                        self._update(transition, kernels)
                    else:
                        logging.warning('Discarding invalid transition "{}"'
                                        .format(transition))

                    break       # XXX Use chained transitions instead.

            # Check to see if we are done. Due to the way MPI works,
            # we can exit after all communication has ceased.
            if remaining == 0:
                if has_converged():
                    if sent == received:
                        break
                else:
                    # We're not done yet. Run more trajectory
                    # fragments and re-check.
                    if free(sent, received) > 0:
                        remaining += free(sent, received)
            else:
                if free(sent, received) > 0:
                    input_transition = distribution.sample()
                    input_transition.remove_files()
                    server.put(input_transition)

                    remaining -= 1
                    sent += 1

        log_progress(sent, received, remaining)

        clock.toc()
        logging.info('Finished sampling on {} [elapsed time: {}].'
                     .format(milestone, clock))

        return kernels

    def _has_converged(self, kernels: Sequence[TransitionKernel],
                       milestone: Milestone, tolerance: float,
                       min_samples: int) -> bool:
        """Criterion to determine convergence at each milestone.

        We keep track of the row of the transition matrix, $K$,
        corresponding to `milestone` as the simulation proceeds.
        Likewise for the matrix of lag times, $T$.  When the two rows
        (regarded as vectors) converge in the $\ell_\infty$ norm and
        we have obtained at least `min_samples`, then we stop the
        simulation.

        Parameters
        ----------
        kernels : Sequence[TransitionKernel]
            Instances of TransitionKernel (one for each parallel process).
        milestone : Milestone
            Milestone from which the trajectory fragments are started.
        tolerance : float
            Threshold for the relative error of the relevant row of
            the transition matrix and the lag time matrix.
        min_samples : int
            Minimum number of samples admissible.

        Returns
        -------
        has_converged : bool
            Whether the convergence criterion has been satisfied.

        """
        def compute_rows(milestone, kernels):
            # XXX This might belong in the matrices module instead.
            M = self.simulation.milestones.max_milestones
            idx = milestone.index

            A = np.zeros((1, M), dtype=np.int32)
            B = np.zeros((1, M), dtype=np.float64)

            for kernel in kernels:
                A += kernel.matrices.A[idx, :]
                B += kernel.matrices.B[idx, :]

            total = np.sum(A)

            K = make_stochastic_matrix(A)
            T = make_time_matrix(A, B)

            Krow = K.toarray()
            Trow = T.toarray()

            return Krow, Trow, total

        has_converged = False

        Kprev = self._prev_Krow
        Tprev = self._prev_Trow

        Krow, Trow, total = compute_rows(milestone, kernels)
        logging.debug('Total for {}: {}'.format(milestone, total))

        if total > 0:
            logging.debug('Row of K corresponding to {} is {}'
                          .format(milestone, vector_to_list(Krow)))
            logging.debug('Row of T corresponding to {} is {}'
                          .format(milestone, vector_to_list(Trow)))

            if Kprev is not None and Tprev is not None:
                Krel_err = norm(Kprev - Krow, np.inf) / norm(Krow, np.inf)
                Trel_err = norm(Tprev - Trow, np.inf) / norm(Trow, np.inf)

                logging.debug('Relative errors with respect to previous '
                              'sample: {:g} (for K) and {:g} (for T).'
                              .format(Krel_err, Trel_err))

                if (Krel_err < tolerance and Trel_err < tolerance
                        and total >= min_samples):
                    # logging.debug('Local convergence has been attained.')
                    has_converged = True

            self._prev_Krow = Krow
            self._prev_Trow = Trow

        return has_converged

    def _update(self, transition: Transition,
                kernels: Sequence[TransitionKernel]) -> None:
        """Update the transition kernel(s) and the database with a new
        transition.

        This method also generates an additional dummy transition if
        necessary.

        """
        logging.debug(transition)
        try:
            self.database.insert_transition(transition, delete_files=True)
        except DatabaseAlreadyExistsError:
            logging.debug('Duplicate database entry for "{}"'
                          .format(transition))
        for kernel in kernels:
            kernel.update(transition)

        # If we reach the product, then generate a dummy transition
        # ending at the reactant.
        if transition.final_milestone in self.products:
            dummy = self.reactant_distribution.sample()
            dummy.lag_time = 0.0
            dummy.initial_milestone = transition.final_milestone
            assert dummy.transition_id
            dummy.remove_files()  # Just in case.
            for kernel in kernels:
                kernel.update(dummy)
