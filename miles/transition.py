"""Transition class.

This module contains the code required to manage transition objects.
These objects contain the information about a trajectory fragment
started at some initial milestone at time zero and stopped at a final
milestone after a certain time (called lag time).  Each transition
object knows how to map the phase space points in the final milestone
as points in the space of collective variables.  This object can also
check the consistency of a transition to discard invalid results.

"""

__all__ = ['Transition', 'TransitionError', 'InitialTransition']

import logging
import os
from typing import List, Optional, Sequence

import numpy as np

from miles import (Milestone, format_colvars, get_random_name)


class TransitionError(Exception):
    """Error constructing transitions.

    """
    pass


class Transition:
    """Transition between two milestones.

    Encapsulates the information of a crossing event between two
    different milestones.

    Attributes
    ----------
    final_milestone : Milestone
        Final milestone.
    initial_milestone : Milestone
        Initial milestone.
    colvars : np.array
        Coordinates of the point in the space of collective variables.
    lag_time : float
        Elapsed time between departing from the initial milestone and
        arriving to the final milestone.
    transition_id : int
        Identifier for the transition as it appears in the database.
    reset_velocities : bool
        Flag indicating whether a trajectory started from the
        end-point of the current transition should reset its velocity
        (i.e., resample from the Maxwell-Boltzmann distribution) or
        continue with the known velocity.
    file_names : List[str]
        List of file names containing auxiliary files for the
        transition. Typically this is comprised of DCD files or other
        files containing phase space points. The latest auxiliary file
        name is placed at the end of the list.

    """
    def __init__(self, final_milestone: Optional[Milestone],
                 initial_milestone: Optional[Milestone],
                 colvars: Optional[np.array],
                 lag_time: Optional[float],
                 transition_id: Optional[int] = None,
                 reset_velocities: bool = False,
                 file_names: Optional[Sequence[str]] = None) -> None:
        self.final_milestone = final_milestone
        self.initial_milestone = initial_milestone
        self.colvars = colvars
        self.lag_time = lag_time
        self.transition_id = transition_id
        self.reset_velocities = reset_velocities
        self.file_names = []     # type: List[str]
        if file_names:
            self.set_files(file_names)

    def set_files(self, file_names: Sequence[str]) -> None:
        """Establish the list of auxiliary files for the transition.

        Parameters
        ----------
        file_names : Sequence[str]
            File names corresponding to the auxiliary files for the
            transition. These files should all have the same prefix
            with different extensions.

        """
        if not file_names:
            return

        if self.file_names:
            self.remove_files()

        self.file_names.extend(file_names)

    def remove_files(self) -> None:
        """Remove auxiliary files from the file system.

        """
        while self.file_names:
            try:
                os.remove(self.file_names.pop())
            except (FileNotFoundError, PermissionError, OSError):
                pass

    def forget(self) -> 'Transition':
        """Forget initial end-point and lag time.

        """
        self.initial_milestone = None
        self.lag_time = None
        return self

    def is_valid(self) -> bool:
        """Determine if transition is well-formed.

        Conduct sanity checks on the transition.

        Returns
        -------
        valid : bool
            Whether the transition is valid or not.

        """
        return (self.initial_milestone
                and self.final_milestone != self.initial_milestone)

    def __del__(self) -> None:
        self.remove_files()

    def __repr__(self) -> str:
        return ('{}({!r}, {!r}, {!r}, {!r}, transition_id={!r}, '
                'reset_velocities={!r}, file_names={!r})'
                .format(self.__class__.__name__,
                        self.final_milestone,
                        self.initial_milestone,
                        self.colvars,
                        self.lag_time,
                        self.transition_id,
                        self.reset_velocities,
                        self.file_names))

    # def __str__(self) -> str:
    #     return 'Transition {}'.format(self.transition_id)
    def __str__(self) -> str:
        if self.initial_milestone is None and self.lag_time is None:
            return ('Transition ending at ({}) in {}'
                    .format(format_colvars(self.colvars),
                            self.final_milestone))
        else:
            assert self.lag_time is not None
            return ('Transition ending at ({}) in {} from {} '
                    'after {:.5g} units of time'
                    .format(format_colvars(self.colvars),
                            self.final_milestone, self.initial_milestone,
                            self.lag_time))

    def __getstate__(self):
        """Prepare the object to be pickled.

        Before pickling a Transition, we create hard links to the
        auxiliary files so that when the object is unpickled, the
        files are available in the file system.

        """
        state = self.__dict__.copy()
        file_names = state['file_names']
        try:
            state['file_names'] = make_hard_links(file_names)
        except FileNotFoundError:
            logging.error('Unable to find {!r} in "{}"'
                          .format(file_names, self))
            raise
        return state


def make_hard_links(file_names: Sequence[str]) -> List[str]:
    """Create hard links with random names."""
    random_name = None

    new_file_names = []

    for file_name in file_names:
        dir_name = os.path.dirname(file_name)
        base_name = os.path.basename(file_name)
        parts = base_name.split(os.path.extsep)
        ext = parts[-1]
        if random_name is None:
            random_name = get_random_name(dir_name)
        new_file_name = os.path.extsep.join([random_name, ext])

        os.link(file_name, new_file_name)

        new_file_names.append(new_file_name)

    return new_file_names


class InitialTransition(Transition):
    """Placeholder transition.

    This class is useful when starting a simulation from an arbitrary
    phase space point that does not necessarily lie in a milestone.

    """
    def __init__(self, reset_velocities: bool) -> None:
        super().__init__(None, None, None, None,
                         reset_velocities=reset_velocities)

    def __str__(self) -> str:
        return ('Transition with initial files {!r} {} velocities'
                .format(', '.join(self.file_names),
                        'resetting' if self.reset_velocities else 'reusing'))

    def set_files(self, file_names: Sequence[str]) -> None:
        new_file_names = make_hard_links(file_names)
        super().set_files(new_file_names)
