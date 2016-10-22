__all__ = ['ChunksNAMD']

import shutil

from typing import Optional

from miles import (BaseChunks, Configuration, get_random_name, delete_files)  # noqa: E501


class ChunksNAMD(BaseChunks):
    """Sequence of NAMD trajectory chunks.

    """
    suffixes = {'coor', 'vel', 'xsc'}

    extra_suffixes = {'colvars.traj', 'colvars.state',
                      'colvars.state.old'}

    def __init__(self, configuration: Configuration,
                 initial_file: str, reset_velocities: bool,
                 random_seed: Optional[int] = None) -> None:
        super().__init__(configuration, initial_file,
                         reset_velocities, random_seed)

    def prepare(self):
        assert self.prev_chunk is not None
        out = self.prev_chunk.output_name
        inp = get_random_name(self.configuration.temp_dir)
        steps = self.configuration.steps_per_chunk

        for ext in self.suffixes:
            old = '{}.{}.{}'.format(out, steps, ext)
            new = '{}.{}'.format(inp, ext)
            shutil.move(old, new)

        return inp

    def cleanup(self):
        if self.num_chunk > 1:
            inp = self.prev_chunk.input_name
            delete_files(self.suffixes, inp)

        out = self.prev_chunk.output_name
        delete_files(self.suffixes.union(self.extra_suffixes), out)

        steps = self.configuration.steps_per_chunk
        for step in range(1, steps + 1):
            file_name = '{}.{}'.format(out, step)
            delete_files(self.suffixes, file_name)
