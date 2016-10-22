"""Orchestrate milestoning simulations."""


__all__ = ['Simulation', 'SimulationError']

import atexit
import logging
import os
import sys
from typing import Optional     # noqa: F401. Used for mypy.

import numpy as np

from miles import (BaseTimestepper, ColvarsParser, Configuration, Database, Distribution, Milestones, TrajectoryParser, load_database, load_distributions, version)  # noqa: E501


class SimulationError(Exception):
    pass


class Simulation:
    """Contains information for a milestoning simulation.

    The contents of this class must not be changed once a simulation
    is running.

    """
    config_file = None
    catch_signals = None
    setup_reactants_and_products = None
    configuration = None
    database = None             # type: Optional[Database]
    milestones = None           # type: Optional[Milestones]
    collective_variables = None
    reactant_distribution = None  # type: Optional[Distribution]

    def __init__(self, config_file: str,
                 catch_signals: bool = True,
                 setup_reactants_and_products: bool = True) -> None:
        """Initialize simulation class.

        Parameters
        ----------
        config_file : str
            Path to the configuration file to be used for the
            simulation.  Environment variables inside the path and
            os-dependent shortcuts for the user's home directory are
            expanded.
        catch_signals : bool, optional
            Whether to set up signal handlers or not. Callers that
            may alter the database should set this option to true.
        setup_reactants_and_products : bool, optional
            Set up reactant and product milestones. Setting this flag
            to false is useful at the early stages when we are about
            to run for the first time in discovery mode.

        """
        print(version.v_gnu)

        sys.setrecursionlimit(10000)

        self.config_file = expand_user_vars(config_file)
        self.configuration = Configuration()
        self.configuration.parse(self.config_file)

        np.set_printoptions(precision=4)

        self.catch_signals = catch_signals
        if self.catch_signals:
            # This set up is process-specific, so we do it to avoid
            # MPI processes from doing it.
            self._setup_signal_handlers()

        self._setup_logging()

        self._setup_collective_variables()

        self._setup_stateful()

        self.setup_reactants_and_products = setup_reactants_and_products
        if self.setup_reactants_and_products:
            self._setup_reactants()
            self._setup_products()

    def __repr__(self) -> str:
        return ('{}({!r}, catch_signals={!r}, '
                'setup_reactants_and_products={!r})'
                .format(self.__class__.__name__,
                        self.config_file, self.catch_signals,
                        self.setup_reactants_and_products))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['database']
        del state['milestones']
        del state['reactant_distribution']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._setup_stateful()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers.

        We enable the necessary signal handlers to ensure the graceful
        termination of a simulation.  Upon arrival of one of the
        masked signals, the database is saved and the program
        exits.

        """
        def exit_handler():
            logging.debug('Saving database.')
            try:
                self.database.save()
            except:
                logging.error('Unable to close database.')

            logging.info('Exiting.')

        atexit.register(exit_handler)

    def _setup_reactants(self) -> None:
        """Set up the distribution at the reactant."""
        dist_file = self.configuration.reactant_distribution
        reactant_distributions = load_distributions(dist_file)

        if not reactant_distributions:
            raise SimulationError('Reactant distribution does not contain '
                                  'points from just one milestone.')

        assert len(reactant_distributions) == 1  # XXX

        for milestone in reactant_distributions:
            self.milestones.append_reactant(milestone)

        milestone, distribution = reactant_distributions.popitem()
        self.reactant_distribution = distribution  # XXX Not future-proof.

    def _setup_products(self) -> None:
        """Set up product milestones."""
        if not self.configuration.product_milestone:
            logging.debug('No product milestones are known.')
            return

        milestones = self.milestones
        pair = self.configuration.product_milestone
        product_milestone = milestones.make_from_indices(pair[0], pair[1])
        milestones.append_product(product_milestone)

    def _setup_logging(self) -> None:
        """Initialize the root logger with the right settings."""
        import logging

        logging.basicConfig(stream=sys.stdout,
                            level=self.configuration.logging_level,
                            format='%(levelname)s: %(message)s')
        logger = logging.getLogger()
        logger.name = 'miles'

    def _setup_collective_variables(self) -> None:
        """Initialize space of collective_variables."""
        colvars_file_name = self.configuration.colvars_file
        if not colvars_file_name:
            raise SimulationError('No colvars input file found')

        colvars_parser = ColvarsParser(colvars_file_name)
        self.collective_variables = colvars_parser.collective_variables

    def _setup_stateful(self):
        """Set up the stateful pieces of the simulation.

        """
        # Load database and set up the collection of milestones.
        self.database = load_database(self.configuration.database_file,
                                      self.collective_variables)
        self.milestones = self.database.milestones

        # Set up temporary directory.
        try:
            temp_dir = self.configuration.temp_dir
            os.mkdir(temp_dir)
        except FileExistsError:
            pass

    def make_trajectory_parser(self) -> TrajectoryParser:
        """Create a TrajectoryParser object.

        """
        return TrajectoryParser(self.milestones, self.configuration,
                                self.collective_variables)

    def make_timestepper(self) -> BaseTimestepper:
        """Create a Timestepper object for the current simulation.

        """
        trajectory_parser = self.make_trajectory_parser()

        command = self.configuration.command

        if 'namd2' in command:
            from miles import TimestepperNAMD
            timestepper_class = TimestepperNAMD
        elif 'brownian-dynamics' in command:
            from miles import TimestepperBD
            timestepper_class = TimestepperBD
        else:
            raise SimulationError('Unknown MD engine: {!r}'.format(command))

        return timestepper_class(trajectory_parser, self.configuration,
                                 self.collective_variables)


def expand_user_vars(file_name: str) -> str:
    """Expand environment variables and user directories.

    """
    return os.path.expandvars(os.path.expanduser(file_name))
