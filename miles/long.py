"""Run long trajectory simulation.

Running simulations by plain Molecular Dynamics is useful for testing
and benchmarking purposes.

"""

__all__ = ['long']

import logging
import math
import random

from miles import (Database, Distribution, Matrices, Milestones, Simulation, TimestepperServer, Transition, TransitionKernel, TransitionKernelError, bold)  # noqa: E501
from miles.analyze import analyze


def long(simulation: Simulation,
         max_trajectories: int,
         num_processes: int) -> None:
    """Run uninterrupted trajectory simulation.

    Parameters
    ----------
    simulation : Simulation
        A concrete simulation object on which to work.
    max_trajectories : int
        Maximum number of reactive trajectories to sample.
    num_processes : int
        Number of MPI processes to use in parallel.

    """
    configuration = simulation.configuration
    for line in str(configuration).split('\n'):
        logging.info(line)

    milestones = simulation.milestones
    log_milestones(milestones)

    # Create a transition kernel.
    matrices = Matrices(simulation.milestones)
    kernel = TransitionKernel(matrices)

    # Reseed the random number generator.
    random.seed(random.randint(1, 65535))

    reactant_distribution = simulation.reactant_distribution

    with TimestepperServer(simulation, num_processes) as timestepper_server:
        try:
            run_long(simulation.database, kernel, max_trajectories,
                     milestones, num_processes, reactant_distribution,
                     timestepper_server)
        except KeyboardInterrupt:
            logging.error('Interrupted.')


def run_long(database, kernel, max_trajectories, milestones,
             num_processes, reactant_distribution, timestepper_server):
    """Collect long (uninterrupted) trajectories.

    """
    mfpt = math.inf

    # Send initial requests.
    sent, received = 0, 0
    for sent in range(min(num_processes, max_trajectories)):
        transition = reactant_distribution.sample()
        timestepper_server.put(transition)

    # Receive transitions and request trajectory continuations.
    num_trajectories = 0
    should_exit = False
    while should_exit is False:
        logging.debug('{}/{}\tsent: {}\treceived: {}\tshould_exit = {}'
                      .format(num_trajectories, max_trajectories,
                              sent, received, should_exit))

        _, output_transitions = timestepper_server.get()
        received += 1

        for transition in output_transitions:
            update(transition, kernel, database)

            if transition.final_milestone in milestones.products:
                logging.debug('"{}" reached the product'.format(transition))
                update_product(transition, kernel, reactant_distribution)

                num_trajectories += 1
                logging.info('Reactive trajectories: {:3d}/{:3d}'
                             .format(num_trajectories, max_trajectories))

                transition = reactant_distribution.sample()
                logging.debug('Starting a new trajectory fragment at "{}"'
                              .format(transition))
                break

        if num_trajectories + sent - received < max_trajectories:
            logging.debug('Continuing from: "{}"'.format(transition))
            timestepper_server.put(transition)
            sent += 1
        elif num_trajectories >= max_trajectories and sent == received:
            should_exit = True

        try:
            mfpt = analyze(kernel, 'stationary.dst', 'K.mtx', 'T.mtx',
                           'q.dat', 't.dat', 'p.dat')
            log_mfpt(mfpt, logging.DEBUG)
        except TransitionKernelError:
            pass

    log_mfpt(mfpt, logging.INFO)


def update(transition: Transition, kernel: TransitionKernel,
           database: Database) -> None:
    """Update transition kernel and database with a new transition.

    In addition to updating the transition kernel and the database, we
    reload the database so that the new transition is immediately
    available for the MPI services to continue it.

    """
    logging.debug('Got "{}"'.format(transition))
    database.insert_transition(transition)
    transition.remove_files()
    kernel.update(transition)


def update_product(transition: Transition,
                   kernel: TransitionKernel,
                   reactant_distribution: Distribution) -> None:
    """Update transition kernel with a transition from product to reactant.

    """
    new_transition = reactant_distribution.sample()
    new_transition.lag_time = 0.0
    new_transition.initial_milestone = transition.final_milestone
    kernel.update(new_transition)


def log_milestones(milestones: Milestones) -> None:
    """Log information about reactants and products.

    """
    def miles_to_str(mls):
        return ','.join([str(m) for m in mls])

    logging.info('Reactant(s): {}'.format(miles_to_str(milestones.reactants)))
    logging.info('Product(s): {}'.format(miles_to_str(milestones.products)))


def log_mfpt(mfpt: float, level: int = logging.INFO) -> None:
    logging.log(level, bold('Mean first passage time: {:.4f}'.format(mfpt)))
