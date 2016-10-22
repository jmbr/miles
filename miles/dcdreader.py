"""Input and output of DCD/DVD trajectory files.

See http://www.ks.uiuc.edu/Research/vmd/plugins/molfile/dcdplugin.html
for a specification of the file format.

"""


__all__ = ['DCDReader', 'DVDReader']

import struct
from io import BytesIO

import numpy as np


# XXX Deal with endianness in DCD files.
# XXX Accelerate with numba?


class DCDReader:
    """Parser for DCD/DVD files."""

    def __init__(self, file_name=None):
        self.file_name = file_name
        self.stream = None
        self.num_particles = None
        self.header = None
        self.frame = None

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.file_name)

    def __str__(self):
        return ('DCD file {!r} for a system with {} particles.'
                .format(self.file_name, self.num_particles))

    def __enter__(self):
        if self.file_name is not None:
            self.stream = open(self.file_name, 'rb')
            self.parse_header()
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stream is not None:
            self.stream.close()

    def parse_header(self):
        self.header = self.stream.read(196)

        f = BytesIO(self.header)

        chunk = f.read(8)
        # array_start = struct.unpack('!i', chunk[:4])
        # print(array_start[0])
        # print(chunk.endswith('CORD'))

        chunk = f.read(80)
        # header = struct.unpack('!' + 'i'*20, chunk)
        # print('Number of frames:', header[0])

        chunk = f.read(4)
        # array_start = struct.unpack('!i', chunk)
        # print(array_start[0])

        chunk = f.read(92)
        # values = struct.unpack('!ii' + 'c'*80 + 'i', chunk)
        # comment = ''.join(values[2:-2])
        # print(comment)

        chunk = f.read(12)
        values = struct.unpack('!iii', chunk)
        self.num_particles = values[1]
        # print(values, self.num_particles)

        f.close()

    def parse_frame(self):
        n = self.num_particles
        chunk_size = 4 + 4 * n + 4

        frame = self.stream.read(chunk_size * 3)
        if len(frame) != chunk_size * 3:
            raise RuntimeError('Invalid frame size.', len(frame))
        else:
            self.frame = frame

        f = BytesIO(frame)

        xyz = np.zeros((n, 3), dtype=np.float32)

        for i in range(3):
            chunk = f.read(4 * n + 8)
            values = struct.unpack('!i' + 'f' * n + 'i', chunk)
            xyz[:, i] = np.array(values[1:-1])

        f.close()

        return xyz

    def parse_frame_num(self, num_frame):
        self.stream.seek(0)
        self.parse_header()
        for n in range(num_frame + 1):
            xyz = self.parse_frame()
        # print('parse_frame_num: fetched frame #', num_frame)
        return xyz

    def save_current_frame_to(self, file_name):
        with open(file_name, 'wb') as f:
            f.write(self.header)
            f.write(self.frame)

    def save_last_frame_to(self, file_name):
        # XXX Replace by seek
        for frame in self:
            pass
        self.save_current_frame_to(file_name)

    def extract_frame_to(self, num_frame, file_name):
        self.parse_frame_num(num_frame)
        self.save_current_frame_to(file_name)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.parse_frame()
        except RuntimeError:
            raise StopIteration

    def __getitem__(self, key):
        """Partial support for self[key]."""
        assert isinstance(key, int)

        for idx, frame in enumerate(self.__iter__()):
            if idx == key:
                return frame


DVDReader = DCDReader
