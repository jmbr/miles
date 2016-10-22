"""Miscellaneous utilities."""


__all__ = ['bold', 'cm2inch', 'colored', 'delete_file', 'delete_files', 'format_colvars', 'get_random_name', 'log_lines', 'log_stack', 'path_ext_join', 'vector_to_list']  # noqa: E501

import errno
import logging
import os
import random
import sys
import traceback
try:
    import colorama
    import termcolor
except ImportError:
    colorama = None
    termcolor = None
from typing import Tuple, List, Optional

import numpy as np

from miles import default_randbits


def bold(msg: str) -> str:
    """Return a string to be printed in bold face."""
    if colorama:
        colorama.init()
        return termcolor.colored(msg, attrs=['bold'])
    else:
        return msg


def colored(msg: str, color: str) -> str:
    """Return a string to be printed in color."""
    if colorama:
        colorama.init()
        return termcolor.colored(msg, color)
    else:
        return msg


def log_lines(msg: bytes, prefix: str = '',
              level: int = logging.DEBUG) -> None:
    """Log each line of a multi-line message."""

    # XXX Replace utf-8 by the best encoding.
    message = msg.decode('utf-8').strip()
    if len(message) == 0:
        return

    for line in message.split('\n'):
        logging.log(level, prefix + line)


def log_stack(log_func=logging.info) -> str:
    """Log the stack frame."""
    frame = sys._getframe().f_back.f_back

    stack_trace = traceback.format_stack(f=frame)
    for line in stack_trace:
        log_func(line.strip())


def path_ext_join(*pieces) -> str:
    """Join pieces of a path adding an extension at the end."""
    return os.extsep.join([os.path.join(*pieces[:-1]),
                           pieces[-1]])


def vector_to_list(v: np.array) -> List[Tuple[int, float]]:
    """Obtain a list of (index, value) where value > 0 from an array."""
    _, jj = v.nonzero()
    return [(j, v[0, j]) for j in jj]


def delete_file(file_name: str) -> None:
    """Delete a file.

    Do not raise an exception if the file does not exist.

    Parameters
    ----------
    file_name
        Path to the file to be deleted.

    """
    try:
        # logging.debug('Deleting file {}.'.format(repr(file_name)))
        os.remove(file_name)
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            raise


def delete_files(suffixes, stem):
    """Delete files starting with stem and ending with any suffix."""
    for ext in suffixes:
        if ext:
            file_name = os.path.extsep.join([stem, ext])
        else:
            file_name = stem
        delete_file(file_name)


def get_random_name(dir_name: Optional[str] = None,
                    randbits: int = default_randbits) -> str:
    """Generate a random file name.

    Parameters
    ----------
    dir_name : str, optional
        Name of directory path preceding the random name.
    randbits : int, optional
        Number of random bits in the generated name.

    Returns
    -------
    path
        Full path of the randomly generated file name.

    """
    path = None

    while True:
        s = hex(random.getrandbits(randbits))

        file_name = s[2:-1]

        if dir_name is not None:
            path = os.path.join(dir_name, file_name)
        else:
            path = file_name

        if not os.path.exists(path):
            break

    return path


def format_colvars(colvars: np.array) -> str:
    """Convert collective variables to string."""
    return ' '.join(format(x, '.4f') for x in colvars)


def cm2inch(*tupl):
    """Convert centimeters to inches."""
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
