"""Module for handling sequences of chunks.

"""

__all__ = ['BaseChunks']

import logging
import random
from abc import ABCMeta, abstractmethod
from typing import Optional

from miles import (Chunk, Clock, Configuration, get_random_name)  # noqa: E501


class BaseChunks(metaclass=ABCMeta):
    """Sequence of trajectory chunks.

    Attributes
    ----------
    num_chunk : int
        Keeps track of the number of instances of Chunk that have been
        used in the Chunks object.
    configuration : Configuration
        Configuration settings for the simulation.
    random_seed : int
        Random seed used for the random number generator (RNG). Some
        of the random numbers generated will in turn be used as seeds
        for the RNGs inside the Molecular Dynamics codes).
    initial_file : str
        Name of the file to use as a starting point for the
        simulation.
    reset_velocities : bool
        Whether to reset velocities or not.
    prev_chunk : Optional[Chunk]
        Latest chunk in the sequence.

    """
    def __init__(self, configuration: Configuration,
                 initial_file: str, reset_velocities: bool,
                 random_seed: Optional[int] = None) -> None:
        self.num_chunk = 0
        self.configuration = configuration
        if random_seed is not None:
            self.random_seed = random_seed
        else:
            self.random_seed = random.randint(1, 65535)
        self.initial_file = initial_file
        self.reset_velocities = reset_velocities
        self.prev_chunk = None  # type: Optional[Chunk]

    def __repr__(self) -> str:
        return ('{}({!r}, {!r}, {!r}, random_seed={!r})'
                .format(self.__class__.__name__,
                        self.configuration,
                        self.initial_file,
                        self.reset_velocities,
                        self.random_seed))

    def make_chunk(self, input_name: str, reset_velocities: bool,
                   random_seed: int) -> Chunk:
        """Instantiate a new chunk."""
        configuration = self.configuration

        output_name = get_random_name(configuration.temp_dir)

        if reset_velocities is True:
            template = configuration.md_template_reset_velocities
        else:
            template = configuration.md_template

        return Chunk(configuration, input_name, output_name,
                     template, random_seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_chunk == 0:
            input_name = self.initial_file
            reset_velocities = self.reset_velocities
            random_seed = self.random_seed
        else:
            input_name = self.prepare()
            reset_velocities = False
            random_seed = self.random_seed + self.num_chunk
            assert self.prev_chunk is not None
            self.cleanup()

        chunk = self.make_chunk(input_name, reset_velocities, random_seed)

        self.num_chunk += 1
        self.prev_chunk = chunk

        logging.debug('Sampling chunk #{}. Reset velocities? {}'
                      .format(self.num_chunk, reset_velocities))

        with Clock() as clock:
            chunk.run()
        logging.debug('chunk.run() completed after {} seconds'.format(clock))

        return chunk

    def __del__(self):
        if self.prev_chunk is not None:
            self.cleanup()

    @abstractmethod
    def prepare(self) -> str:
        """Prepare files for a new iteration/chunk.

        This method takes care of copying the last frame from the
        previous chunk to a new set of files that will become the
        first frame of the next chunk.

        Returns
        -------

        prefix : str
            Prefix for the file name of the new set of files.

        """
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        """Delete temporary files.

        Delete temporary files generated during the previous chunk.

        """
        raise NotImplementedError
