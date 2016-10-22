"""MPI process for servicing time-stepper requests."""


__all__ = ['timestepper_service']

import logging
import random
import sys
import time
from typing import Optional, Sequence

from mpi4py import MPI

from miles import Clock, DatabaseFileNotFoundError, Simulation, Transition, log_lines  # noqa: E501


MAX_ATTEMPTS = 20


class TimestepperServiceError(Exception):
    """Timestepper service error."""
    pass


class TimestepperService:
    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation
        self.timestepper = simulation.make_timestepper()
        self.database = simulation.database
        self.temp_dir = self.simulation.configuration.temp_dir

    def sample(self, transition: Transition) -> Sequence[Transition]:
        """Sample transitions initiated from a given point.

        Parameters
        ----------
        transition : Transition
            Transition to be used as a starting point for a new
            trajectory fragment. It is only necessary that
            transition.file_name be initialized, all the remaining
            attributes may be set to None.

        Returns
        -------
        transitions : Sequence[Transition]
            A sequence of transitions. The reason why it is a sequence
            is that after running a single chunk, several transitions
            may be observed, in which case they are used as part of
            the computation.

        Raises
        ------
        TimestepperServiceError in case there is a problem extracting
        the auxiliary files for the transition.

        """
        attempts = 1
        while True:
            try:
                with Clock() as clock:
                    self.database.extract_files(transition, self.temp_dir)

                transitions = self.timestepper.run(transition)

                logging.debug('database.extract_files completed after '
                              '{} seconds.'.format(clock))

                return transitions
            except:
                if attempts >= MAX_ATTEMPTS:
                    log_error(sys.exc_info())
                    raise TimestepperServiceError('Unable to run "{}"'
                                                  .format(transition))

                logging.info('Attempting file extraction ({}/{})'
                             .format(attempts, MAX_ATTEMPTS))
                self.database.reload_zip_file_for_reading()
                time.sleep(random.random())
                attempts += 1


class Port:
    """Context manager for MPI ports."""
    def __init__(self, service_name, rank):
        self.service_name = service_name
        self.rank = rank
        self.port = None

    def __enter__(self):
        assert self.port is None
        self.port = MPI.Open_port()
        MPI.Publish_name(self.service_name, self.port)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.port is not None:
            MPI.Unpublish_name(self.service_name, self.port)
            MPI.Close_port(self.port)


def setup_random_seed() -> int:
    """Set up seed for the random number generator.

    Returns
    -------

    seed : int
        Independent random seed for each rank in the process group.

    """
    rank = MPI.COMM_WORLD.Get_rank()

    seed = int(random.randint(0, sys.maxsize) +
               hash(MPI.Get_processor_name()) + rank)

    random.seed(seed)

    return seed


def timestepper_service() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logging.basicConfig(level=logging.DEBUG)
    service_name = 'server-{:04d}'.format(rank)
    logger = logging.getLogger(service_name)

    setup_random_seed()

    if rank == 0:
        print('Starting MPI services...', flush=True)

    with Port(service_name, rank) as port:
        wait_and_notify(comm, service_name, port,
                        'Ready. Run simulation with:\n  '
                        'mpirun -n 1 miles run --mpi-processes {} ...'
                        .format(size),
                        '{service_name}: Listening on port {port!r}.')

        logger = logging.getLogger()
        logger.name = service_name

        comm_self = MPI.COMM_SELF
        icomm = comm_self.Accept(port.port)

        wait_and_notify(comm, service_name, port,
                        'All MPI services are connected.',
                        '{service_name}: Connected.')

        simulation = icomm.recv()
        simulation.database.reload_zip_file_for_reading()

        timestepper_service = TimestepperService(simulation)

        while True:
            transition = icomm.recv()
            if transition is None:
                logger.info('Received exit signal.')
                break

            logger.info('Received "{}"'.format(transition))

            try:
                observed_transitions = timestepper_service.sample(transition)
            except TimestepperServiceError as exc:
                logging.error('Unable to sample "{}". Continuing...'
                              .format(transition))
                observed_transitions = []

            if not observed_transitions:
                logging.error('Sending back empty observed transitions')
            output_transitions = (transition, observed_transitions)

            logger.info('Sending {} transitions corresponding to "{}".'
                        .format(len(observed_transitions), transition))
            icomm.send(output_transitions, dest=0)
            # logger.info('Results sent.')

        icomm.Disconnect()

        wait_and_notify(comm, service_name, port,
                        'All MPI services have been disconnected.',
                        '{service_name}: Disconnected.')


def wait_and_notify(comm: MPI.Comm, service_name: str, port: Port,
                    message: str, message_per_rank: Optional[str] = None) \
        -> None:
    """Wait until a barrier is reached and then notify the user.

    """
    comm.barrier()

    rank = comm.Get_rank()
    size = comm.Get_size()

    data = service_name, port.port
    data = comm.gather(data, root=0)

    if rank == 0:
        for i in range(size):
            if message_per_rank:
                info = {'service_name': data[i][0],
                        'port': data[i][1]}
                print(message_per_rank.format(**info))
        print(message, flush=True)


def log_error(exc_info):
    import traceback
    exc_msg = traceback.format_exception(*exc_info)

    for line in exc_msg:
        logging.warning(line[:-1])
