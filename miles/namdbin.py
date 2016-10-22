"""Module for loading and writing files in namdbin format.

"""


__all__ = ['NAMDBinError', 'NAMDBinReader', 'NAMDBinWriter', 'NAMDBinSequence', 'read_namdbin', 'save_namdbin']  # noqa: E501

import itertools
import struct

import numpy as np


class NAMDBinError(Exception):
    pass


class NAMDBinReader:
    """Read files in namdbin format.

    This class reads files in namdbin format (typically used for
    restarts) into Numpy arrays.

    """
    def __init__(self, file_name):
        self.file_name = file_name
        self.stream = None
        self.num_particles = None

    def _open(self):
        self.stream = open(self.file_name, 'rb')
        n = struct.unpack('i', self.stream.read(4))
        self.num_particles = n[0]
        if self.num_particles <= 0:
            raise NAMDBinError('Invalid number of particles: {}.'
                               .format(self.num_particles))
        return self

    def __enter__(self):
        return self._open()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stream is not None:
            self.stream.close()

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__,
                                 repr(self.file_name))

    def read(self):
        xyz = np.fromfile(self.stream, dtype=np.float64,
                          count=3*self.num_particles)
        return xyz.reshape((self.num_particles, 3))

    def __del__(self):
        if self.stream is not None:
            self.stream.close()


class NAMDBinWriter:
    """Write files to namdbin format.

    This class writes a numpy array to namdbin format.

    """
    def __init__(self, file_name):
        self.file_name = file_name
        self.stream = None
        self.num_particles = None

    def _open(self):
        self.stream = open(self.file_name, 'wb')
        return self

    def __enter__(self):
        return self._open()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stream is not None:
            self.stream.close()

    def write(self, xyz):
        num_atoms = xyz.shape[0]
        self.stream.write(struct.pack('i', num_atoms))
        return xyz.tofile(self.stream)

    def __del__(self):
        if self.stream is not None:
            self.stream.close()


class NAMDBinSequence:
    def __init__(self, prefix, suffix):
        self.prefix = prefix
        self.suffix = suffix

    def __iter__(self):
        for n in itertools.count(1):
            file_name = '{}{}{}'.format(self.prefix, n, self.suffix)
            try:
                with NAMDBinReader(file_name) as namdbin:
                    yield namdbin.read()
            except FileNotFoundError:
                return


def save_namdbin(xyz, file_name):
    """Save a numpy array in namdbin format.

    Parameters
    ----------
    xyz : np.array
        Array of dimensions N x 3, where N is the number of atoms.
    file_name : str
        Path to the file where the data will be saved.

    """
    with NAMDBinWriter(file_name) as f:
        return f.write(xyz)


def read_namdbin(file_name: str) -> np.array:
    """Read NAMDBin file.

    Parameters
    ----------
    file_name : str
        File name to read.

    Returns
    -------
    xyz : np.array
        Array of coordinates. Its dimension is N x 3 where N is the
        number of atoms.

    """
    with NAMDBinReader(file_name) as f:
        return f.read()
