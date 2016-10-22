#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK                                -*- mode: python -*-

"""Command line interface (CLI) for the milestoning tool."""

import argparse
try:
    import argcomplete
except ImportError:
    argcomplete = None
import faulthandler
import random
import signal
import sys
import time

import miles.commands as commands
from miles.profiler import Profiler


def main():
    parser = argparse.ArgumentParser(description='Milestoning tool')
    parser.add_argument('--profile', required=False, metavar='FILE',
                        type=argparse.FileType('w', encoding='utf-8'),
                        help='run under profiler and save report to '
                        '%(metavar)s')
    parser.add_argument('--random-seed', required=False,
                        metavar='RANDOM-SEED', type=float,
                        default=time.time(), help='use prescribed '
                        'random seed')

    subparsers = parser.add_subparsers(help='sub-command help', dest='cmd')

    cmds = commands.Commands(subparsers, commands.command_list)

    if argcomplete:
        argcomplete.autocomplete(parser)

    args = parser.parse_args(sys.argv[1:])

    random.seed(args.random_seed)

    if args.cmd is None:
        parser.print_help()
        sys.exit(-1)

    faulthandler.register(signal.SIGUSR1)

    cmd = cmds[args.cmd]
    with Profiler(args.profile):
        cmd.do(args)

    if args.profile:
        args.profile.close()


if __name__ == '__main__':
    main()
