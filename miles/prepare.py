"""Obtain phase space points at a given milestone as a prerequisite
for running a milestoning simulation.

"""


__all__ = ['prepare']

import copy
import logging
import os
import random
from typing import List, Optional, Set

from miles import (InitialTransition, Milestone, Simulation, TimestepperNAMD, TrajectoryParser)  # noqa: E501


def prepare(simulation: Simulation,
            milestone: Optional[Milestone],
            input_file: str,
            start: int, stop: int, step: int) -> None:
    """Run a trajectory until a number of transitions are observed.

    This function runs a trajectory started at a given phase space
    point. Initially, the trajectory runs indefinitely until it hits a
    target milestone. From then on, a subset of the subsequent points
    crossing that milestone are added to the database.

    Parameters
    ----------
    simulation : Simulation
        The simulation object pointing to the database, the
        configuration, and the milestone factory.
    milestone : Milestone, optional
        The milestone(s) that we want to sample points from.
    input_file : str
        Name of initial file to use in order to start the simulation.
    start : int
        Number of points to skip before starting to save transitions
        to the database.
    stop : int
        Maximum number of points in the target milestone to consider
        for storage.
    step : int
        Number of points in the target milestone to skip between
        successive storage into the database.

    """
    assert 0 <= start <= stop and step > 0

    database = simulation.database
    milestones = simulation.milestones
    configuration = simulation.configuration
    collective_variables = simulation.collective_variables

    trajectory_parser = TrajectoryParser(milestones, configuration,
                                         collective_variables)
    trajectory_parser.collect_crossing_events = True

    initial_transition = InitialTransition(reset_velocities=True)
    initial_transition.set_files(get_initial_files(input_file))

    timestepper = TimestepperNAMD(trajectory_parser, configuration,
                                  collective_variables)

    known_milestones = set()

    should_stop = False
    current = 1
    while True:
        if should_stop:
            break

        random_seed = random.randint(1, 65535)
        transitions = timestepper.run(initial_transition, random_seed)
        assert transitions

        to_insert = []

        for transition in transitions:
            if milestone and (transition.final_milestone != milestone or
                              transition.initial_milestone is not None):
                continue

            known_milestones.add(transition.final_milestone)

            if start <= current and current % step == 0:
                transition.forget()
                logging.debug(transition)
                to_insert.append(transition)
                known_milestones.add(transition.final_milestone)
                report_progress(current, stop, milestone, known_milestones)

            current += 1

            if current > stop:
                should_stop = True

        initial_transition = copy.copy(transition)
        database.insert_transitions(to_insert, delete_files=True)


def report_progress(current: int, stop: int, milestone: Milestone,
                    known_milestones: Set[Milestone]) -> None:
    if len(known_milestones) < 5:
        milestones_str = milestones_to_str(known_milestones, milestone)
        logging.info('Got {}/{} phase space points on {}'
                     .format(current, stop, milestones_str))
    else:
        logging.info('Got {}/{} phase space points on {} milestones'
                     .format(current, stop, len(known_milestones)))


def milestones_to_str(milestones: Set[Milestone],
                      milestone: Milestone) -> str:
    if milestone is not None:
        return str(milestone)
    else:
        return ', '.join([str(m) for m in milestones])


def get_initial_files(initial_file: str) -> List[str]:
    # Remove extension, then add suffixes.
    parts = initial_file.split(os.extsep)
    stem = parts[0]
    return [os.extsep.join([stem, ext])
            for ext in ('coor', 'vel', 'xsc')]  # XXX NAMD-specific
