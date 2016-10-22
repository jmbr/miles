__all__ = ['TimestepperBD']

import os

import numpy as np

import miles.default as default
from miles import (BaseTimestepper, ChunksBD, get_random_name)


class TimestepperBD(BaseTimestepper):
    """Runs a trajectory from an initial phase space point until it hits a
    milestone.

    """
    def __init__(self, trajectory_parser, configuration,
                 max_chunks=default.max_chunks):
        super().__init__(trajectory_parser, configuration, max_chunks)

        self.chunks_class = ChunksBD

    def find_transitions_in_file(self, file_name):
        """Find transitions in an output file."""
        data = np.loadtxt(file_name)
        return self.trajectory_parser.parse(data[:, 1:3])

    def find_transitions(self, chunk):
        """Find transitions in a chunk."""
        # We store the transitions in a list. The alternative would be
        # to turn this into a generator method but this complicates
        # matters when sending the transitions via MPI.
        transitions = self.find_transitions_in_file(chunk.output_name)

        for transition in transitions:
            self.save(transition)

        return transitions

    def save(self, transition):
        """Store files associated to a transition."""
        temp_dir = self.configuration.temp_dir
        temp_file_name = get_random_name()
        transition.file_name = os.path.join(temp_dir, temp_file_name)

        x, y = transition.colvars
        with open(transition.file_name, 'w') as f:
            print('{:f} {:f}'.format(x, y), file=f)
