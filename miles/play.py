"""Inspection of trajectory files in search for transitions between
milestones.

"""


__all__ = ['play']

import itertools
import logging
import os
import sys
from typing import List, Optional


from miles import (Milestone, Simulation, Transition)  # noqa: E501


ALLOWED_SUFFIXES = {'coor', 'vel', 'xsc'}


def play(simulation: Simulation, milestone: Optional[Milestone],
         file_name: str, collect_crossing_events: bool,
         start: int, stop: int, step: int) -> None:
    """Find transitions in trajectory files.

    Parameters
    ----------
    simulation : Simulation
        Simulation object used to instantiate a timestepper, which in
        turn is used to extract transitions.
    milestone : Optional[Milestone]
        Target milestone. All transitions that do not end at this
        milestone will be discarded. If this parameter is set to None,
        it means that every transition will be considered.
    file_name : str
        Name of one of the files belonging to the sequence of phase
        space points to be analyzed.
    collect_crossing_events : bool
        Whether to discard crossing events and consider transitions
        between neighboring milestones only.
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
    if not os.path.exists(file_name):
        logging.error('File not found {!r}'.format(file_name))
        sys.exit(-1)

    if not is_allowed_file(file_name):
        logging.error('Cannot inspect {!r}. File format not implemented'
                      .format(file_name))
        sys.exit(-1)

    transitions = find_transitions(simulation, file_name,
                                   collect_crossing_events)

    if not transitions:
        logging.info('No transitions found')
    else:
        count = insert_transitions(simulation, transitions, milestone,
                                   start, step, stop)
        logging.info('Saved {} out of {} transition(s)'
                     .format(count, len(transitions)))


def find_transitions(simulation: Simulation, file_name: str,
                     collect_crossing_events: bool) -> List[Transition]:
    """Parse transitions in a sequence of files.

    """
    prefix = get_prefix(file_name)

    timestepper = simulation.make_timestepper()

    trajectory_parser = timestepper.trajectory_parser
    trajectory_parser.setup()
    trajectory_parser.collect_crossing_events = collect_crossing_events

    transitions = timestepper.find_transitions(prefix)

    return transitions


def insert_transitions(simulation: Simulation, transitions: List[Transition],
                       milestone: Optional[Milestone], start: int, step: int,
                       stop: int) -> int:
    """Insert transitions into database.

    """
    database = simulation.database

    if milestone:
        valid_transitions = [t for t in transitions
                             if t.final_milestone == milestone
                             and t.is_valid()]
    else:
        valid_transitions = [t for t in transitions if t.is_valid()]

    if stop == float('inf'):
        stop = len(valid_transitions)

    count = 0
    for transition in itertools.islice(valid_transitions, start, stop, step):
        transition.forget()
        logging.info(transition)
        database.insert_transition(transition, delete_files=True)
        count += 1

    return count


def get_prefix(file_name: str) -> str:
    """Get file name prefix.

    """
    parts = file_name.split(os.path.extsep)
    return os.path.extsep.join(parts[:-2])


def is_allowed_file(file_name: str) -> bool:
    """Determine if a given file name can be parsed.

    """
    parts = file_name.split(os.path.extsep)

    return parts[-1] in ALLOWED_SUFFIXES
