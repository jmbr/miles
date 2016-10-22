"""Management of iterations in a milestoning simulation."""


__all__ = ['Iteration']

import logging
import time
from typing import Tuple, Optional

import numpy as np

from miles import (Clock, Distributions, Matrices, Milestone, Simulation, TimestepperServer, TransitionKernel, TransitionSampler, bold)  # noqa: E501


class Iteration:
    """An iteration of Milestoning.

    Each iteration generates its own empty instance of
    TransitionKernel that is populated as the simulation proceeds.

    """
    def __init__(self, simulation: Simulation,
                 kernel: TransitionKernel,
                 initial_distributions: Distributions,
                 min_samples: int, tolerance: float, number: int,
                 timestepper_server: TimestepperServer) -> None:
        self.simulation = simulation
        self.global_transition_kernel = kernel
        self.initial_distributions = initial_distributions
        self.min_samples = min_samples
        self.tolerance = tolerance
        self.milestones = simulation.milestones

        # Create a fresh transition kernel for this particular
        # iteration.
        matrices = Matrices(self.milestones)
        self.transition_kernel = TransitionKernel(matrices)

        self.timestepper_server = timestepper_server
        self.transition_sampler = TransitionSampler(simulation,
                                                    timestepper_server)

        self.number = number

        self.products = set(self.milestones.products)  # type: Set[Milestone]

    def __repr__(self) -> str:
        return ('{}(simulation={!r}, kernel={!r}, '
                'initial_distributions={!r}, min_samples={!r}, '
                'tolerance={!r}, number={!r}, timestepper_server={!r})'
                .format(self.__class__.__name__, self.simulation,
                        self.transition_kernel, self.initial_distributions,
                        self.min_samples, self.tolerance, self.number,
                        self.timestepper_server))

    def run(self, prev_q: Optional[np.array] = None) \
            -> Tuple[Distributions, np.array]:
        """Run a single iteration.


        This method loops over the milestones, picking up the
        corresponding initial distribution and running trajectory
        fragments (i.e., obtaining transitions) started at the points
        drawn from each initial distribution. Each observed transition
        is used to populate the TransitionKernel object.

        After enough transitions are collected, we estimate the
        stationary distribution.

        Parameters
        ----------
        prev_q : np.array, optional
            Normalized vector of weights from previous iteration.

        Returns
        -------
        distributions : Distributions
            A suitable linear combination of first hitting point
            distributions to be used in the next iteration.
        q : np.array
            Normalized vector of weights of each distribution.

        Raises
        ------
        TransitionKernelError
            In case it is not possible to compute the initial
            distributions for the next iteration.

        """
        sampler = self.transition_sampler
        initial_distributions = self.initial_distributions

        empty_milestones = {m for m, d in
                            initial_distributions.items() if len(d) == 0}
        products = self.products
        excluded = products.union(empty_milestones)

        milestones = set(initial_distributions) - excluded
        remaining_milestones = milestones.copy()

        kernel = self.transition_kernel
        kernels = [kernel, self.global_transition_kernel]
        min_samples = self.min_samples
        tolerance = self.tolerance

        database = self.simulation.database

        clock = Clock()
        clock.tic()

        while remaining_milestones:
            milestone = remaining_milestones.pop()
            distribution = initial_distributions[milestone]

            logging.info(bold('Sampling at {} from a pool of {} '
                              'points on {}. Remaining: {} milestones.'
                              .format(milestone, len(distribution),
                                      time.ctime(time.time()),
                                      len(remaining_milestones))))

            kernels = sampler.sample(min_samples, milestone,
                                     distribution, kernels, tolerance)

            database.save()

        clock.toc()
        logging.info('Sampling concluded at {} [elapsed time: {}].'
                     .format(time.ctime(time.time()), clock))

        distributions, q = kernel.compute_distributions(prev_q)

        return distributions, q
