__all__ = ['TimestepperNAMD']

import logging
import os
import shutil
from typing import List, Optional, Union

import numpy as np

import miles.default as default
from miles import (BaseTimestepper, Chunk, ChunksNAMD, CollectiveVariables, Clock, Configuration, Mapping, PhaseSpacePoint, Transition, TrajectoryParser, get_random_name, read_namdbin)  # noqa: E501


class TimestepperNAMD(BaseTimestepper):
    """Timestepper using NAMD2.

    """
    def __init__(self, trajectory_parser: TrajectoryParser,
                 configuration: Configuration,
                 collective_variables: CollectiveVariables,
                 max_chunks: int = default.max_chunks) -> None:
        super().__init__(trajectory_parser, configuration,
                         collective_variables, max_chunks)

        self.chunks_class = ChunksNAMD
        self.mapping = None     # type: Optional[Mapping]
        self.steps_per_chunk = self.configuration.steps_per_chunk

    def find_transitions(self, chunk_or_file_name: Union[Chunk, str]) \
            -> List[Transition]:
        """Find transitions within sequences of NAMD binary files.

        """
        if isinstance(chunk_or_file_name, Chunk):
            file_name = chunk_or_file_name.output_name
        else:
            file_name = chunk_or_file_name

        with Clock() as clock:
            colvars = self._get_colvars_from_colvars_traj(file_name)
        logging.debug('get_colvars_from_colvars_traj completed after '
                      '{} seconds'.format(clock))

        # if self.collective_variables.indices:
        #     colvars = self._get_colvars_from_colvars_traj(file_name)
        # else:
        #     colvars = self._get_colvars_from_phase_space_point(file_name)

        return self.find_transitions_in_colvars(colvars, file_name)

    def _get_colvars_from_colvars_traj(self, file_name: str) -> np.array:
        full_file_name = os.path.extsep.join([file_name, 'colvars.traj'])
        colvars = np.loadtxt(full_file_name)

        steps_per_chunk = self.configuration.steps_per_chunk

        # Ignore the first line which corresponds to the phase space
        # point used as initial condition (i.e., before the first time
        # step).
        assert colvars.shape[0] == steps_per_chunk + 1
        return colvars[1:steps_per_chunk, self.collective_variables.indices]

    # def _get_colvars_from_phase_space_point(self, file_name: str) -> np.array:
    #     N, D = self.configuration.steps_per_chunk, self.mapping.colvars_dim
    #
    #     # We first read all but the last frame in the trajectory. Then
    #     # we read the last frame. This is due to the way NAMD stores
    #     # trajectories.
    #     colvars = np.empty((N, D), dtype=np.float64)
    #
    #     for n in range(N - 1):
    #         colvars[n, :] = self._read_colvars(file_name, n + 1)
    #     colvars[-1, :] = self._read_colvars(file_name)
    #
    #     return colvars

    def _read_colvars(self, file_name: str, n: Optional[int] = None) \
            -> np.array:
        """Read phase space point and return collective variables.

        """
        extsep = os.path.extsep

        if n:
            name = extsep.join([file_name, str(n)])
        else:
            name = file_name

        x = read_namdbin(extsep.join([name, 'coor']))
        # v = read_namdbin(extsep.join([name, 'vel']))
        # p = PhaseSpacePoint(x, v, self.mapping)
        p = PhaseSpacePoint(x, None, self.mapping)

        return p.colvars

    def find_transitions_in_colvars(self, colvars: np.array,
                                    prefix: str) -> List[Transition]:
        """Find transitions within sequences of collective variables.

        """
        with Clock() as clock:
            pairs = self.trajectory_parser.parse(colvars)
        logging.debug('trajectory_parser.parse completed after {} seconds'
                      .format(clock))

        transitions = []
        with Clock() as clock:
            for n, transition in pairs:
                if n == self.steps_per_chunk:
                    input_file_name = prefix
                else:
                    input_file_name = '{}.{}'.format(prefix, n+1)
                self.save(transition, input_file_name)
                transitions.append(transition)
        logging.debug('timestepper.save completed after {} seconds'
                      .format(clock))

        return transitions

    def save(self, transition: Transition, output_name: str) -> None:
        """Store files associated to a transition into the database.

        """
        output_dir = self.configuration.simulation_dir
        file_name = os.path.join(output_dir, get_random_name())

        dest_file_names = []
        for suffix in ('coor', 'vel', 'xsc'):
            orig_file_name = full_file_name(output_name, suffix)
            dest_file_name = full_file_name(file_name, suffix)
            shutil.copy(orig_file_name, dest_file_name)
            dest_file_names.append(dest_file_name)

        transition.set_files(dest_file_names)


def full_file_name(name: str, suffix: str) -> str:
    """Return a full file name.

    """
    return os.extsep.join([name, suffix])
