__all__ = ['ChunksBD']

import os
from typing import Optional

from miles import (BaseChunks, Configuration, delete_files, get_random_name)


class ChunksBD(BaseChunks):
    """Sequence of simple-brownian-dynamics trajectory chunks.

    """
    def __init__(self, configuration: Configuration,
                 initial_file: str, reset_velocities: bool,
                 random_seed: Optional[int] = None) -> None:
        super().__init__(configuration, initial_file,
                         reset_velocities, random_seed)

        self.suffixes = ('',)
        self.intermediate_files = []  # type: List[str]

    def prepare(self):
        assert self.prev_chunk is not None
        out = self.prev_chunk.output_name

        inp = get_random_name(self.configuration.temp_dir)
        self.intermediate_files.append(inp)

        with open(out, 'r') as src, open(inp, 'w') as dst:
            lines = src.readlines()
            last_line = lines[-1].strip()
            fields = last_line.split()
            dst.write('{} {}'.format(fields[1], fields[2]))

        return inp

    def cleanup(self):
        delete_files(self.suffixes, self.prev_chunk.output_name)

    def __del__(self):
        for f in self.intermediate_files:
            os.remove(f)

        super().__del__()
