"""Commands for the user interface."""

__all__ = [name for name in dir() if name.lower().startswith('Command')]

import argparse
try:
    import argcomplete
except ImportError:
    argcomplete = None
import logging
import sys
from abc import ABCMeta, abstractmethod
from typing import Sequence

import miles.default as default
from miles import (Configuration, Distributions, Milestones, Simulation, bold, colored, load_database, load_distributions, make_database, save_distributions, version)  # noqa: E501


class Command(metaclass=ABCMeta):
    """A command."""

    name = None                 # type: str

    @abstractmethod
    def setup_parser(self, subparsers):
        raise NotImplementedError

    @abstractmethod
    def do(self, args):
        raise NotImplementedError

    @staticmethod
    def add_argument_config(parser):
        arg = parser.add_argument('config', type=str,
                                  metavar='CONFIG-FILE', help='name of '
                                  'the configuration file')
        if argcomplete:
            completer = argcomplete.completers.FilesCompleter
            arg.completer = completer(allowednames='cfg')


class Commands:
    """A collection of commands."""

    def __init__(self, parser, commands):
        self.commands = []

        for command in commands:
            command.setup_parser(parser)
            self.commands.append(command)

    def __getitem__(self, command_name):
        for command in self.commands:
            if command.name == command_name:
                return command


def filter_distributions(milestones: Milestones,
                         milestones_str: Sequence[str],
                         distributions: Distributions) \
        -> Distributions:
    """Select specific distributions."""
    if milestones_str:
        selected_distributions = Distributions()

        for milestone_str in milestones_str:
            milestone = milestones.make_from_str(milestone_str)
            if milestone in distributions.keys():
                selected_distributions[milestone] = distributions[milestone]

        return selected_distributions
    else:
        return distributions


class CommandRun(Command):
    """Run milestoning simulation"""

    name = 'run'

    def setup_parser(self, subparsers):
        """Command-line options for exact milestoning."""
        description = self.__doc__
        p = subparsers.add_parser(self.name, help=description.lower(),
                                  description=description.capitalize())
        self.add_argument_config(p)
        b = p.add_argument('-i', '--input', type=str, required=True,
                           metavar='FILE', help='path to '
                           'file containing initial distributions')
        if argcomplete:
            completer = argcomplete.completers.FilesCompleter
            b.completer = completer(allowednames='dst')
        p.add_argument('-s', '--samples', type=int,
                       default=default.samples_per_milestone_per_iteration,
                       metavar='MIN-SAMPLES', help='minimum number of '
                       'trajectory fragments to sample per milestone '
                       'per iteration (default is %(default)d)')
        p.add_argument('-l', '--local-tolerance', type=float,
                       default=default.local_convergence_tolerance,
                       help='tolerance for convergence within each milestone '
                       '(default is %(default)g)')
        p.add_argument('-g', '--global-tolerance', type=float,
                       default=default.global_convergence_tolerance,
                       help='tolerance for convergence of the '
                       'iterative process (default is %(default)g)')
        p.add_argument('-m', '--max-iterations', type=int,
                       metavar='MAX-ITERATIONS',
                       default=default.max_iterations, help='maximum '
                       'number of iterations (default is '
                       '%(default)d)')
        p.add_argument('-p', '--include-products',
                       action='store_true', default=False,
                       help='sample trajectories from product '
                       'milestones (default is %(default)s)')
        p.add_argument('-M', '--mpi-processes', type=int,
                       required=True, help='number of MPI processes')

    def do(self, args):
        simulation = Simulation(args.config)

        try:
            distributions = load_distributions(args.input)
        except FileNotFoundError:
            logging.error('Unable to find {!r}'.format(args.input))
            sys.exit(-1)

        # Remove product milestones from the initial distributions.
        if args.include_products is False:
            for milestone in simulation.milestones.products:
                if milestone in distributions.keys():
                    del distributions[milestone]

        from miles.run import run
        run(simulation, initial_distributions=distributions,
            num_iterations=args.max_iterations,
            num_samples=args.samples,
            local_tolerance=args.local_tolerance,
            global_tolerance=args.global_tolerance,
            num_processes=args.mpi_processes)


class CommandSample(Command):
    """Sample trajectory fragments on specific milestones"""

    name = 'sample'

    def setup_parser(self, subparsers):
        """Command-line options for exact milestoning."""
        description = self.__doc__
        p = subparsers.add_parser(self.name, help=description.lower(),
                                  description=description.capitalize())
        self.add_argument_config(p)
        b = p.add_argument('-i', '--input', type=str, required=True,
                           metavar='FILE', help='path to '
                           'file containing initial distributions')
        if argcomplete:
            completer = argcomplete.completers.FilesCompleter
            b.completer = completer(allowednames='dst')
        p.add_argument('-m', '--milestone', metavar='MILESTONE',
                       required=True, action='append', help='restrict'
                       ' to specified milestone(s)')
        p.add_argument('-s', '--samples', type=int,
                       default=default.samples_per_milestone_per_iteration,
                       metavar='MIN-SAMPLES', help='minimum number of '
                       'trajectory fragments to sample per milestone '
                       'per iteration (default is %(default)d)')
        p.add_argument('-l', '--local-tolerance', type=float,
                       default=default.local_convergence_tolerance,
                       help='tolerance for convergence within each milestone '
                       '(default is %(default)g)')
        p.add_argument('-M', '--mpi-processes', type=int,
                       required=True, help='number of MPI processes')

    def do(self, args):
        simulation = Simulation(args.config)

        try:
            initial_distributions = load_distributions(args.input)
        except FileNotFoundError:
            logging.error('Unable to find {!r}'.format(args.input))
            sys.exit(-1)

        distributions = filter_distributions(simulation.milestones,
                                             args.milestone,
                                             initial_distributions)

        if len(distributions) == 0:
            logging.error('{!r} does not contain points for {}.'
                          .format(args.input,
                                  ','.join(args.milestone)))
            sys.exit(-1)

        from miles.run import run
        run(simulation,
            initial_distributions=distributions,
            num_iterations=1,
            num_samples=args.samples,
            local_tolerance=args.local_tolerance,
            global_tolerance=default.global_convergence_tolerance,
            num_processes=args.mpi_processes)


class CommandPlot(Command):
    """Plot results"""

    name = 'plot'

    def setup_parser(self, subparsers):
        """Command-line options for plotting results."""
        description = self.__doc__
        p = subparsers.add_parser(self.name, help=description.lower(),
                                  description=description.capitalize())
        self.add_argument_config(p)

        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument('-v', '--voronoi', action='store_true',
                       help='plot Voronoi tessellation (default is '
                       '%(default)s)')
        g.add_argument('-H', '--histograms', action='store_true',
                       default=False, help='plot histograms of '
                       'distributions at each milestone')
        p.add_argument('-l', '--labels', action='store_true',
                       default=False, help='plot milestone indices '
                       '(default is %(default)s)')
        p.add_argument('-n', '--num-bins', required=False,
                       help='number of bins for histogram (default '
                       'is %(default)s)')
        p.add_argument('-s', '--marker-size', type=int, required=False,
                       default=default.marker_size, help='set marker size '
                       '(default is %(default)d)')
        b = p.add_argument('-c', '--colors', type=str, required=False,
                           default=default.colors, help='specify color '
                           'scheme (default is %(default)s)')
        if argcomplete:
            import matplotlib.cm as cm
            color_maps = list(cm.datad.keys())
            completer = argcomplete.completers.ChoicesCompleter
            b.completer = completer(color_maps)
        p.add_argument('-t', '--title', type=str, required=False,
                       help='set title')
        p.add_argument('-b', '--colorbar-title', type=str,
                       metavar='CB-TITLE', required=False,
                       help='set color bar title')
        p.add_argument('-m', '--min-value', type=float, required=False,
                       default=0.0, help='set lower bound for data')
        p.add_argument('-M', '--max-value', type=float, required=False,
                       default=None, help='set upper bound for data')
        p.add_argument('-x', '--xlabel', type=str, required=False,
                       default='$x_1$', help='set label of x axis')
        p.add_argument('-y', '--ylabel', type=str, required=False,
                       default='$x_2$', help='set label of y axis')
        a = p.add_argument('-i', '--input', type=str,  # required=True,
                           metavar='FILE', action='append',
                           help='path(s) to input data file(s)')
        if argcomplete:
            a.completer = argcomplete.completers.FilesCompleter()
        p.add_argument('-o', '--output', type=str, required=False,
                       metavar='FILE', help='output figure name')

    def do(self, args):
        simulation = Simulation(args.config, catch_signals=False,
                                setup_reactants_and_products=False)

        if args.title is None and args.input is not None:
            args.title = args.input[0]

        from miles.plot import plot
        plot(simulation, **args.__dict__)


class CommandLong(Command):
    """Run long trajectory simulation"""

    name = 'long'

    def setup_parser(self, subparsers):
        """Command-line options for long trajectories."""
        description = self.__doc__
        p = subparsers.add_parser(self.name, help=description.lower(),
                                  description=description.capitalize())
        self.add_argument_config(p)
        p.add_argument('-i', '--interval', type=int, metavar='N',
                       default=default.save_interval, required=False,
                       help='save results to disk every %(metavar)s '
                       'transitions (default is %(default)d)')
        p.add_argument('-m', '--max-trajectories', type=int,
                       required=False, metavar='MAX-TRAJECTORIES',
                       default=default.trajectory_max_trajectories,
                       help='maximum number of transitions to '
                       'sample (default is %(default)d)')
        p.add_argument('-M', '--mpi-processes', type=int,
                       required=True, help='number of MPI processes')

    def do(self, args):
        simulation = Simulation(args.config)

        from miles.long import long
        long(simulation, max_trajectories=args.max_trajectories,
             num_processes=args.mpi_processes)


class CommandMkDist(Command):
    """Create distribution file from database"""

    name = 'mkdist'

    def setup_parser(self, subparsers):
        """Command line options for creation of distribution files."""
        description = self.__doc__
        p = subparsers.add_parser(self.name, help=description.lower(),
                                  description=description.capitalize())
        self.add_argument_config(p)
        p.add_argument('-d', '--database', type=str,
                       metavar='DIRECTORY', required=False,
                       help='directory where the database is located '
                       '(default is set to the path specified in the '
                       'configuration file)')
        p.add_argument('-m', '--milestone', metavar='MILESTONE',
                       action='append', help='restrict to specified '
                       'milestone(s)')
        p.add_argument('-o', '--output', type=str,
                       metavar='FILE', required=True,
                       help='name of the file where the distributions '
                       'will be written')
        p.add_argument('-l', '--less-than', type=int, metavar='NUM-POINTS',
                       required=False, help='maximum number of points that a'
                       'milestone should have')
        p.add_argument('-g', '--greater-than', type=int, metavar='NUM-POINTS',
                       required=False, help='minimum number of points that a'
                       'milestone should have')
        group = p.add_mutually_exclusive_group(required=False)
        group.add_argument('-r', '--reset-velocities',
                           action='store_true', dest='reset_velocities',
                           default=True, required=False, help='set initial '
                           'velocities from Maxwell-Boltzmann distribution '
                           '(default is %(default)s)')
        group.add_argument('-n', '--no-reset-velocities',
                           action='store_false', dest='reset_velocities',
                           default=False, required=False, help='do not set '
                           'initial velocities from Maxwell-Boltzmann '
                           'distribution (default is %(default)s)')

    def do(self, args):
        simulation = Simulation(args.config, catch_signals=False,
                                setup_reactants_and_products=False)

        milestones = simulation.milestones

        if args.database is not None:
            collective_variables = simulation.collective_variables
            db = load_database(args.database, collective_variables)
        else:
            db = simulation.database

        ds = db.to_distributions(args.reset_velocities)

        new_ds = Distributions()

        restricted_milestones = set()
        if args.milestone:
            for mls in args.milestone:
                restricted_milestone = milestones.make_from_str(mls)
                restricted_milestones.add(restricted_milestone)

        for milestone in ds.keys():
            if args.milestone and milestone not in restricted_milestones:
                continue

            d = ds[milestone]
            l = len(d)

            if not ((args.greater_than and l < args.greater_than)
                    or (args.less_than and l > args.less_than)):
                new_ds[milestone] = d
                print('{} has {} points'.format(milestone, l))

        save_distributions(new_ds, args.output)

        num_distributions = len(list(new_ds.keys()))

        logging.info('Wrote {} distribution(s) to {!r}.'
                     .format(num_distributions, args.output))


class CommandLsDist(Command):
    """Print information about a set of distributions"""

    name = 'lsdist'

    def setup_parser(self, subparsers):
        description = self.__doc__
        p = subparsers.add_parser(self.name, help=description.lower(),
                                  description=description.capitalize())
        self.add_argument_config(p)
        a = p.add_argument('-i', '--input', type=str, required=True,
                           metavar='FILE', help='path to file '
                           'containing distributions')
        if argcomplete:
            completer = argcomplete.completers.FilesCompleter
            a.completer = completer(allowednames='dst')

    def do(self, args):
        try:  # Attempt to use reactants and products if there are any.
            simulation = Simulation(args.config, catch_signals=False)
        except:
            simulation = Simulation(args.config, catch_signals=False,
                                    setup_reactants_and_products=False)

        milestones = simulation.milestones

        try:
            distributions = load_distributions(args.input)
        except FileNotFoundError:
            logging.error('Unable to find {!r}'.format(args.input))
            sys.exit(-1)

        known_milestones = sorted(distributions.keys())
        for milestone in known_milestones:
            distribution = distributions[milestone]
            msg = ('{} has {} points'
                   .format(milestone, len(distribution)))

            if milestone in milestones.reactants:
                print(colored(bold(' '.join([msg, '(reactant)'])), 'red'))
            elif milestone in milestones.products:
                print(colored(bold(' '.join([msg, '(product)'])), 'green'))
            else:
                print(msg)


class CommandPath(Command):
    """Compute max-weight path"""

    name = 'path'

    example = '''
Example
-------
The command
    miles path simulation.cfg --reactant 0,1 --product 2,3 --product 3,4 \\
               --transition-matrix K.mtx --stationary-vector q.dat
will compute the maximum weight path between the milestone 0,1 (reactant)
and milestones 2,3 or 3,4 (products).
'''

    def setup_parser(self, subparsers):
        description = self.__doc__
        p = subparsers.add_parser(self.name, help=description.lower(),
                                  epilog=self.example,
                                  formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=description.capitalize())
        self.add_argument_config(p)
        p.add_argument('-K', '--transition-matrix', type=str,
                       metavar='FILE', required=True,
                       help='file name of transition matrix')
        p.add_argument('-q', '--stationary-vector', type=str,
                       metavar='FILE', required=True,
                       help='file name of stationary vector')
        p.add_argument('-r', '--reactant', metavar='MILESTONE',
                       action='append', help='use specified milestone(s) '
                       'as reactant')
        p.add_argument('-p', '--product', metavar='MILESTONE',
                       action='append', help='use specified milestone(s) '
                       'as product')
        p.add_argument('-o', '--output', metavar='FILE', type=str,
                       default='path.dat', help='file name where '
                       'to save the maximum weight path (default '
                       'is %(default)s)')

    def do(self, args):
        import scipy.io
        import numpy as np
        from miles.max_weight_path import max_weight_path

        simulation = Simulation(args.config, catch_signals=False)

        K = scipy.io.mmread(args.transition_matrix).tocoo()
        q = np.loadtxt(args.stationary_vector)

        milestones = simulation.milestones

        if args.reactant:
            reactant_indices = {milestones.make_from_str(a).index
                                for a in args.reactant}
        else:
            reactant_indices = {m.index for m in milestones.reactants}

        if args.product:
            product_indices = {milestones.make_from_str(a).index
                               for a in args.product}
        else:
            product_indices = {m.index for m in milestones.products}

        print('Reactant milestones:')
        for idx in reactant_indices:
            print('  {}'.format(milestones.make_from_index(idx)))
        print('Product milestones:')
        for idx in product_indices:
            print('  {}'.format(milestones.make_from_index(idx)))

        path = max_weight_path(K, q, reactant_indices, product_indices)
        np.savetxt(args.output, path)

        print('Maximum weight path written to {!r}'.format(args.output))


class CommandCite(Command):
    """Obtain citations of relevant papers"""

    name = 'cite'

    def setup_parser(self, subparsers):
        description = self.__doc__
        subparsers.add_parser(self.name, help=description.lower(),
                              description=description.capitalize())

    def do(self, args):
        bibtex_citation = bold('The exact milestoning algorithm is '
                               'described in:') + r"""
@article{Bello-Rivas2015,
author = {Bello-Rivas, J. M. and Elber, R.},
doi = {10.1063/1.4913399},
issn = {0021-9606},
journal = {The Journal of Chemical Physics},
month = {mar},
number = {9},
pages = {094102},
title = {{Exact milestoning}},
url = {https://doi.org/10.1063/1.4913399},
volume = {142},
year = {2015}
}

""" + \
    bold('The algorithm for the computation of global '
         'max-weight paths is described in:') + r"""
@article{Viswanath2013,
author = {Viswanath, S. and Kreuzer, S. M. and
          Cardenas, A. E. and Elber, R.},
doi = {10.1063/1.4827495},
issn = {00219606},
journal = {The Journal of Chemical Physics},
month = {nov},
number = {17},
pages = {174105},
title = {{Analyzing milestoning networks for molecular kinetics:
          definitions, algorithms, and examples}},
url = {https://doi.org/10.1063/1.4827495},
volume = {139},
year = {2013}
}"""
        print(bibtex_citation)


DISCLAIMER = ("""{} the stationary distributions obtained by the {}
and {} commands are only valid for databases generated by the {}
command.""".format(bold('Warning:'), bold('analyze'),
                   bold('resample'), bold('long')))


class CommandAnalyze(Command):
    """Analyze results and generate output files"""

    name = 'analyze'

    def setup_parser(self, subparsers):
        description = self.__doc__
        p = subparsers.add_parser(self.name, help=description.lower(),
                                  epilog=DISCLAIMER,
                                  description=description.capitalize())
        self.add_argument_config(p)
        p.add_argument('-o', '--output', metavar='FILE',
                       type=str, default='stationary-analyze.dst',
                       help='file name where to save the stationary '
                       'distributions (default is %(default)s)')
        p.add_argument('-K', '--transition-matrix', type=str,
                       metavar='FILE', default='K.mtx',
                       help='file name of the transition matrix '
                       '(default is %(default)s)')
        p.add_argument('-T', '--lag-time-matrix', type=str,
                       metavar='FILE', default='T.mtx',
                       help='file name of the lag time matrix '
                       '(default is %(default)s)')
        p.add_argument('-q', '--stationary-flux', type=str,
                       metavar='FILE', default='q.dat',
                       help='file name of the stationary flux vector '
                       '(default is %(default)s)')
        p.add_argument('-t', '--local-mfpts', type=str,
                       metavar='FILE', default='t.dat',
                       help='file name of the vector of local MFPTs '
                       '(default is %(default)s)')
        p.add_argument('-p', '--stationary-probability', type=str,
                       metavar='FILE', default='p.dat',
                       help='file name of the stationary probability vector '
                       '(default is %(default)s)')

    def do(self, args):
        simulation = Simulation(args.config, catch_signals=False)

        from miles.analyze import TransitionKernel, analyze
        kernel = TransitionKernel(simulation.database)
        mfpt = analyze(kernel, args.output, args.transition_matrix,
                       args.lag_time_matrix, args.stationary_flux,
                       args.local_mfpts, args.stationary_probability)

        logging.info('Mean first passage time: {:.4f} units of time.'
                     .format(mfpt))


class CommandCommittor(Command):
    """Compute the committor function"""

    name = 'committor'

    def setup_parser(self, subparsers):
        description = self.__doc__
        p = subparsers.add_parser(self.name, help=description.lower(),
                                  description=description.capitalize())
        self.add_argument_config(p)
        p.add_argument('-K', '--transition-matrix', type=str,
                       metavar='FILE', required=True,
                       help='file name of the transition matrix')
        p.add_argument('-o', '--output', metavar='FILE', type=str,
                       default='committor.dat', help='file name where '
                       'to save the committor vector (default '
                       'is %(default)s)')

    def do(self, args):
        import scipy.io
        import numpy as np
        from miles.committor import committor

        simulation = Simulation(args.config, catch_signals=False)

        K = scipy.io.mmread(args.transition_matrix).tocsr()

        milestones = simulation.milestones
        reactant_indices = {m.index for m in milestones.reactants}
        product_indices = {m.index for m in milestones.products}

        committor_vector = committor(K, reactant_indices, product_indices)
        np.savetxt(args.output, committor_vector)

        print('Committor function written to {!r}'.format(args.output))


class CommandResample(Command):
    """Resample database and analyze results"""

    name = 'resample'

    def setup_parser(self, subparsers):
        description = self.__doc__
        p = subparsers.add_parser(self.name, help=description.lower(),
                                  epilog=DISCLAIMER,
                                  description=description.capitalize())
        self.add_argument_config(p)
        p.add_argument('-s', '--samples', type=int,
                       default=10*default.samples_per_milestone_per_iteration,
                       metavar='NUM-SAMPLES', help='number of '
                       'trajectory fragments per milestone to '
                       'sample (default is %(default)d)')
        p.add_argument('-d', '--database', metavar='DIRECTORY',
                       type=str, default='database-resample',
                       help='output database directory name '
                       '(default is %(default)s)')
        p.add_argument('-o', '--output', metavar='FILE', type=str,
                       default='stationary-resample.dst', help='file '
                       'name where to save the resampled stationary '
                       'distributions (default is %(default)s)')
        p.add_argument('-K', '--transition-matrix', type=str,
                       metavar='FILE', default='K-resample.mtx',
                       help='file name of the transition matrix '
                       '(default is %(default)s)')
        p.add_argument('-T', '--lag-time-matrix', type=str,
                       metavar='FILE', default='T-resample.mtx',
                       help='file name of the lag time matrix '
                       '(default is %(default)s)')
        p.add_argument('-q', '--stationary-flux', type=str,
                       metavar='FILE', default='q-resample.dat',
                       help='file name of the stationary flux vector '
                       '(default is %(default)s)')
        p.add_argument('-t', '--local-mfpts', type=str,
                       metavar='FILE', default='t-resample.dat',
                       help='file name of the vector of local MFPTs '
                       '(default is %(default)s)')
        p.add_argument('-p', '--stationary-probability', type=str,
                       metavar='FILE', default='p.dat',
                       help='file name of the stationary probability vector '
                       '(default is %(default)s)')

    def do(self, args):
        simulation = Simulation(args.config, catch_signals=False)

        from miles.resample import resample
        resample(simulation, args.samples, args.database, args.output,
                 args.transition_matrix, args.lag_time_matrix,
                 args.stationary_flux, args.local_mfpts,
                 args.stationary_probability)


class CommandVersion(Command):
    """Display program's version"""

    name = 'version'

    def setup_parser(self, subparsers):
        description = self.__doc__
        subparsers.add_parser(self.name, help=description.lower(),
                              description=description.capitalize())

    def do(self, args):
        print(bold(version.v_gnu))


class CommandReset(Command):
    """Reset velocities in a distribution"""

    name = 'reset'

    def setup_parser(self, subparsers):
        description = self.__doc__
        p = subparsers.add_parser(self.name, help=description.lower(),
                                  description=description.capitalize())
        a = p.add_argument('-i', '--input', type=str, required=True,
                           metavar='FILE', help='path to '
                           'file containing distributions')
        if argcomplete:
            completer = argcomplete.completers.FilesCompleter
            a.completer = completer(allowednames='dst')
        p.add_argument('-o', '--output', type=str, required=True,
                       metavar='FILE', help='path to file '
                       'where to save the transformed distributions')

    def _reset_velocities(self, ds):
        pass

    def do(self, args):
        try:
            distributions = load_distributions(args.input)
        except FileNotFoundError:
            logging.error('Unable to find {!r}'.format(args.input))
            sys.exit(-1)

        self._reset_velocities(distributions)

        save_distributions(distributions, args.output)


class CommandMPI(Command):
    """Start MPI services

    """
    name = 'mpi'

    def setup_parser(self, subparsers):
        description = self.__doc__.strip()
        subparsers.add_parser(self.name, help=description.lower(),
                              description=description.capitalize())

    def do(self, args):
        from miles.timestepper_service import timestepper_service
        timestepper_service()


class CommandPrepare(Command):
    """Sample first hitting points

    """
    name = 'prepare'

    _suffixes = ('coor', 'vel', 'xsc', 'dcd', 'dvd', 'dcdvel')

    def setup_parser(self, subparsers):
        description = self.__doc__.strip()
        p = subparsers.add_parser(self.name, help=description.lower(),
                                  description=description.capitalize())
        self.add_argument_config(p)
        a = p.add_argument('-i', '--input', type=str, required=True,
                           metavar='FILE', help='path to file '
                           'containing initial results for the MD program')
        if argcomplete:
            completer = argcomplete.completers.FilesCompleter
            a.completer = completer(allowednames=self._suffixes)

        p.add_argument('-m', '--milestone', metavar='MILESTONE',
                       required=False, help='restrict to specified milestone')
        p.add_argument('--start', type=int, required=False,
                       metavar='NUM-CROSSINGS', default=0,
                       help='number of milestone crossings to obtain before '
                       'starting to save to database (default is %(default)s)')
        p.add_argument('--stop', type=int, required=True,
                       metavar='NUM-CROSSINGS', help='maximum number of '
                       'milestone crossings to obtain')
        p.add_argument('--step', type=int, required=False,
                       metavar='NUM-CROSSINGS', default=1,
                       help='number of milestone crossings to skip (default '
                       'is %(default)s)')

    def do(self, args):
        simulation = Simulation(args.config, catch_signals=True,
                                setup_reactants_and_products=False)

        from miles.prepare import prepare

        if args.milestone:
            milestone = simulation.milestones.make_from_str(args.milestone)
        else:
            milestone = None

        prepare(simulation, milestone, args.input, args.start,
                args.stop, args.step)


class CommandPlay(Command):
    """Find hitting points in trajectory files

    """
    name = 'play'

    _suffixes = ('coor', 'vel', 'xsc', 'dcd', 'dvd', 'dcdvel')

    def setup_parser(self, subparsers):
        description = self.__doc__.strip()
        p = subparsers.add_parser(self.name, help=description.lower(),
                                  description=description.capitalize())
        self.add_argument_config(p)
        a = p.add_argument('-i', '--input', type=str, required=True,
                           metavar='FILE', help='path to file '
                           'containing initial results for the MD program')
        if argcomplete:
            completer = argcomplete.completers.FilesCompleter
            a.completer = completer(allowednames=self._suffixes)

        p.add_argument('-m', '--milestone', metavar='MILESTONE',
                       required=False, help='restrict to specified milestone')
        p.add_argument('-c', '--crossings', required=False, default=False,
                       action='store_true', help='include same-milestone '
                       'crossings in addition to transitions between '
                       'neighboring milestones (default is %(default)s)')
        p.add_argument('--start', type=int, required=False,
                       metavar='NUM-CROSSINGS', default=0,
                       help='number of milestone crossings to obtain before '
                       'starting to save to database (default is %(default)s)')
        p.add_argument('--stop', type=int, required=False,
                       default=float('inf'), metavar='NUM-CROSSINGS',
                       help='maximum number of milestone crossings to obtain'
                       ' (default is %(default)s)')
        p.add_argument('--step', type=int, required=False,
                       metavar='NUM-CROSSINGS', default=1,
                       help='number of milestone crossings to skip (default '
                       'is %(default)s)')

    def do(self, args):
        simulation = Simulation(args.config, catch_signals=True,
                                setup_reactants_and_products=False)

        from miles.play import play

        if args.milestone:
            milestone = simulation.milestones.make_from_str(args.milestone)
        else:
            milestone = None

        play(simulation, milestone, args.input, args.crossings,
             args.start, args.stop, args.step)


class CommandMkdb(Command):
    """Create database of first hitting points"""

    name = 'mkdb'

    def setup_parser(self, subparsers):
        description = self.__doc__
        p = subparsers.add_parser(self.name, help=description.lower(),
                                  description=description.capitalize())
        self.add_argument_config(p)
        a = p.add_argument('-a', '--anchors', type=str, required=True,
                           metavar='FILE', help='path to file '
                           'containing anchors in CSV format')
        t = p.add_argument('-t', '--transitions', type=str,
                           metavar='FILE', help='path to file '
                           'containing transitions in CSV format')
        if argcomplete:
            completer = argcomplete.completers.FilesCompleter
            a.completer = completer(allowednames='csv')
            t.completer = completer(allowednames='csv')

    def do(self, args):
        config = Configuration()
        config.parse(args.config)
        from miles import ColvarsParser

        colvars_parser = ColvarsParser(config.colvars_file)
        print('Using collective variables defined in {!r}.'
              .format(config.colvars_file))
        print('Please verify that the collective variables shown '
              'below are correct.\n')
        print(colvars_parser)

        collective_variables = colvars_parser.collective_variables

        database = make_database(config.database_file, collective_variables)

        database.insert_anchors_from_csv(args.anchors)

        if args.transitions is not None:
            database.insert_transitions_from_csv(args.transitions)


class CommandFsck(Command):
    """Verify database integrity"""

    name = 'fsck'

    def setup_parser(self, subparsers):
        description = self.__doc__
        p = subparsers.add_parser(self.name, help=description.lower(),
                                  description=description.capitalize())
        self.add_argument_config(p)

    def do(self, args):
        simulation = Simulation(args.config, catch_signals=False)

        from miles.fsck import fsck
        status = fsck(simulation)
        sys.exit(status)


command_list = [CommandMkdb(),
                CommandMkDist(),
                CommandLsDist(),
                CommandPrepare(),
                CommandPlay(),
                CommandMPI(),
                CommandRun(),
                CommandPlot(),
                CommandLong(),
                CommandAnalyze(),
                CommandCommittor(),
                CommandPath(),
                CommandFsck(),
                CommandVersion(),
                CommandCite(),
                # CommandResample(),
                # CommandReset(),
                # CommandSample(),
                ]
