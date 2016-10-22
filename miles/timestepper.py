"""Generate trajectory fragments and spot transitions in them.

"""


__all__ = ['BaseTimestepper', 'TimestepperError']  # noqa: E501

import itertools
import logging
import os
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Union

import numpy as np

import miles.default as default

from miles import (BaseChunks, Chunk, CollectiveVariables, Configuration, TrajectoryParser, Transition, bold, log_lines)  # noqa: E501


class TimestepperError(Exception):
    pass


class BaseTimestepper(metaclass=ABCMeta):
    """Run a trajectory until it hits a neighboring milestone.

    """
    def __init__(self, trajectory_parser: TrajectoryParser,
                 configuration: Configuration,
                 collective_variables: CollectiveVariables,
                 max_chunks: int = default.max_chunks) -> None:
        self.trajectory_parser = trajectory_parser
        self.configuration = configuration
        self.collective_variables = collective_variables
        self.max_chunks = max_chunks
        self.chunks = None      # type: Optional[Sequence[BaseChunks]]
        self.chunks_class = None

    def __repr__(self) -> str:
        return ('{}({!r}, {!r}, {!r}, max_chunks={!r})'
                .format(self.__class__.__name__,
                        self.trajectory_parser,
                        self.configuration,
                        self.collective_variables,
                        self.max_chunks))

    def run(self, transition: Transition,
            random_seed: Optional[int] = None) -> Sequence[Transition]:
        """Run trajectory fragment.

        This method runs a trajectory fragment starting from the
        endpoint of `transition` until at least another transition is
        found among the resulting time steps.

        Parameters
        ----------
        transition : Transition
            Transition to be used as a starting point for a new
            trajectory fragment. It is only necessary that
            transition.file_names contains valid entries, all the
            remaining attributes may be set to None.
        random_seed : int, optional
            Seed for the random number generator.

        Returns
        -------
        transitions : Sequence[Transition]
            A sequence of transitions. The reason why it is a sequence
            is that after running a single chunk, several transitions
            may be observed, in which case they are used as part of
            the computation.

        """
        # Initialize reporter.
        self.trajectory_parser.setup(transition.colvars,
                                     transition.final_milestone)

        # Initialize sequence of chunks.
        assert transition.file_names, \
            'No file names associated to "{}"'.format(transition)
        first_file_name = transition.file_names[0]
        inputname = first_file_name.split(os.extsep)[0]

        reset_velocities = transition.reset_velocities

        # We store the Chunks object in Timestepper so that it does
        # not go out of scope and gets garbage-collected before the
        # user can do something about the transitions.
        self.chunks = self.chunks_class(self.configuration, inputname,
                                        reset_velocities, random_seed)

        transitions = []        # type: List[Transition]

        for chunk in itertools.islice(self.chunks, self.max_chunks):
            # log_lines(chunk.stdout)

            if chunk.status != 0:
                logging.error(bold('Problem running chunk with command "{}".'
                                   ' Exit status: {}.'
                                   .format(chunk.command_line[-1],
                                           chunk.status)))
                log_lines(chunk.stdout, prefix='stdout: ', level=logging.ERROR)
                log_lines(chunk.stderr, prefix='stderr: ', level=logging.ERROR)

                raise TimestepperError('Unsuccessful chunk. Exit status: {}'
                                       .format(chunk.status))

            transitions = self.find_transitions(chunk)
            if transitions:
                break

        return transitions

    @abstractmethod
    def find_transitions(self, chunk_or_file_name: Union[Chunk, str]) -> Sequence[Transition]:
        """Find transitions in a chunk's output.

        """
        raise NotImplementedError

    @abstractmethod
    def find_transitions_in_colvars(self, colvars: np.array, prefix: str) \
            -> Sequence[Transition]:
        """Find transitions in a sequence of collective variables.

        """
        raise NotImplementedError

    @abstractmethod
    def save(self, transition: Transition, *args, **kwargs) \
            -> None:
        """Store files associated to a transition into the database.

        """
        raise NotImplementedError
