"""Server side / load balancer side of the parallel client/server
architecture.

"""


__all__ = ['TimestepperServer', 'TimestepperServerError']

import logging
import queue
import threading
import time
from typing import Tuple, List, Optional

from mpi4py import MPI

from miles import (Simulation, Transition)


SERVICE_FORMAT = 'server-{:04d}'

POLL_INTERVAL = 0.01


class TimestepperServerError(Exception):
    """Exception for problems related to MPI communication."""
    pass


class TimestepperServer:
    """Load balancer for simultaneous timestepping on many computers.

    Attributes
    ----------
    simulation : Simulation
        Simulation object containing specific parameters and auxiliary
        structures.
    num_servers : int
        Number of servers available.
    comm : MPI.Intracomm
        Main MPI communicator.
    icomms : List[MPI.Intracomm]
        List of MPI intercommunicators. Each intercommunicator is
        connected to a server.
    input_queue : queue.Queue
        Queue where incoming transitions are placed.
    output_queue : queue.Queue
        Queue where processed transitions are stored.
    memoize : bool
        Whether input transitions should be kept in a cache for reuse
        or not.
    cache : Dict[int, List[Transition]]
        Cache of memoized transitions indexed by `transition_id`.
    thread : Thread
        Processor thread. This thread polls the servers to find new
        results.

    """
    def __init__(self, simulation: Simulation, num_servers: int) -> None:
        self.simulation = simulation
        self.num_servers = num_servers
        self.comm = MPI.COMM_WORLD  # type: MPI.Intracomm
        self.icomms = []            # type: List[MPI.Intracomm]
        self.input_queue = queue.Queue()  # type: queue.Queue
        self.output_queue = queue.Queue()  # type: queue.Queue
        self.memoize = self.simulation.configuration.memoize
        self.cache = {}         # type: Dict[int, Transition]
        self.thread = None      # type: Optional[threading.Thread]
        self._stop = threading.Event()

    def __repr__(self) -> str:
        return ('{}({!r}, {!r})'
                .format(self.__class__.__name__,
                        self.simulation,
                        self.num_servers))

    def connect(self) -> None:
        """Connect to servers.

        Connect to each of the available servers and send the
        simulation object.

        Raises
        ------
        TimestepperServerError
            In case something goes wrong.

        """
        try:
            for num_server in range(self.num_servers):
                service_name = SERVICE_FORMAT.format(num_server)
                try:
                    port = MPI.Lookup_name(service_name)
                    logging.debug('Connecting to {} {}'
                                  .format(service_name, port))
                except MPI.Exception as exc:
                    logging.error('Unable to find {!r}. Exception: {}'
                                  .format(service_name, exc))
                    raise
                icomm = self.comm.Connect(port)
                self.icomms.append(icomm)

                logging.debug('Connected to {}'.format(service_name))

                try:
                    icomm.send(self.simulation, dest=0)
                except MPI.Exception:
                    logging.error('Unable to establish communication with {}'
                                  .format(service_name))
                    raise
        except:
            logging.error('Unable to establish connection to servers.')
            raise TimestepperServerError

        # Start a separate thread for the server.
        self.thread = threading.Thread(target=self._loop)
        self.thread.start()

    def _service_name(self, icomm: MPI.Intracomm) -> str:
        assert self.icomms
        return SERVICE_FORMAT.format(self.icomms.index(icomm))

    def stop(self) -> None:
        """Order the threads to stop.

        """
        self._stop.set()

    def stopped(self) -> bool:
        """Determine if the threads have been signalled to stop.

        """
        return self._stop.is_set()

    def disconnect(self) -> None:
        """Shut down servers.

        Signals the servers to shut down.

        """
        for _ in self.icomms:
            self.put(None)

        if self.thread:
            self.thread.join()
            self.thread = None

        for icomm in self.icomms:
            service_name = self._service_name(icomm)
            logging.debug('Disconnecting from {}'.format(service_name))
            icomm.Disconnect()
            logging.debug('Disconnected from {}'.format(service_name))

    def _memoize(self, input_transition: Transition,
                 output_transitions: List[Transition]) -> None:
        """Memoize transition.

        Save the transition to the cache if appropriate.

        Parameters
        ----------
        input_transition : Transition
            Input transition
        output_transitions : List[Transition]
            Transition(s) resulting from continuing the trajectory
            started at the end-point of `input_transition`.

        Notes
        -----
        Transitions whose `reset_velocities` attribute is true will
        not be memoized.

        """
        if not self.memoize or input_transition.reset_velocities:
            return

        cache = self.cache
        tid = input_transition.transition_id
        assert isinstance(tid, int)
        cache[tid] = output_transitions[0]

    def _recall(self, transition: Transition) -> bool:
        """Recall a transition if it has been seen before.

        Look up a transition in the cache and reuse its result if it
        is already known.

        Parameters
        ----------
        transition : Transition
            Transition to recall.

        Returns
        -------
        recalled : bool
            Whether `transition` was already in the cache or not.

        Notes
        -----
        If `transition` is already known, its result will be enqueued
        in `output_queue`.

        """
        if not self.memoize or not transition:
            return False

        cache = self.cache
        tid = transition.transition_id
        if tid in cache:
            assert not transition.reset_velocities
            output_transition = cache[tid]
            logging.debug('Resampling: "{}" -> "{}"'
                          .format(transition, output_transition))
            self.output_queue.put((transition, [output_transition]))
            return True
        else:
            return False

    def put(self, transition: Optional[Transition]) -> None:
        """Place a new transition in the work queue.

        Parameters
        ----------
        transition : Optional[Transition]
            A transition to use as a starting point for a trajectory
            fragment. When a `transition` is `None` this signals that
            the MPI connections should be dismantled.

        """
        recalled = self._recall(transition)
        if not recalled:
            self.input_queue.put(transition)

    def get(self, block=True) -> Optional[Tuple[Transition, List[Transition]]]:
        """Retrieve transitions from the output queue.

        Parameters
        ----------
        block : bool
            If True, then wait for a result to be ready. Otherwise
            return immediately regardless of the availability of a
            result.

        Returns
        -------
        result : Optional[Tuple[Transition, List[Transition]]]
            If a result is available, then return a tuple containing
            the initial transition and by a list of subsequent
            transitions coming from the trajectory fragment started at
            the initial transition. If `block` is True and no result
            is available, return None.

        """
        try:
            result = self.output_queue.get(block)
            self._memoize(*result)
        except queue.Empty:
            result = None

        return result

    def _loop(self) -> None:
        """Process requests and fetch results.

        """
        requests_pending = []   # type: List[MPI.Request]

        available_icomms = self.icomms.copy()
        used_icomms = []

        while True:
            try:
                transition = self.input_queue.get_nowait()
                logging.debug('Got "{}" from input queue.'
                              .format(transition))
                work_pending = True
            except queue.Empty:
                work_pending = False

            if available_icomms and work_pending and not self.stopped():
                # We have work to do and workers available.
                icomm = available_icomms.pop()

                if transition is None:
                    self._send_exit_signal(icomm)
                else:
                    service_name = self._service_name(icomm)
                    logging.debug('Sending "{}" to {}'
                                  .format(transition, service_name))
                    icomm.send(transition, dest=0)
                    request = icomm.irecv()
                    requests_pending.append(request)
                    used_icomms.append(icomm)
                    logging.debug('Sent "{}" to {}'
                                  .format(transition, service_name))

            if requests_pending:
                index, flag, data = MPI.Request.testany(requests_pending)
                if index == MPI.UNDEFINED:
                    time.sleep(POLL_INTERVAL)
                    continue

                icomm = used_icomms[index]

                transition, observed_transitions = data[0], data[1]
                self.input_queue.task_done()
                self.output_queue.put((transition, observed_transitions))
                self.output_queue.task_done()

                logging.debug('Received {} transitions from {}'
                              .format(len(observed_transitions),
                                      self._service_name(icomm)))

                available_icomms.append(icomm)
                del requests_pending[index]
                del used_icomms[index]

            if (not requests_pending
                    and (not available_icomms or self.stopped())):
                while available_icomms:
                    icomm = available_icomms.pop()
                    self._send_exit_signal(icomm)
                break

            time.sleep(POLL_INTERVAL)

    def _send_exit_signal(self, icomm: MPI.Intracomm) -> None:
        """Send exit signal to intercommunicator.

        """
        service_name = self._service_name(icomm)
        logging.debug('Sending exit signal to {}:{}'
                      .format(icomm, service_name))

        icomm.send(None, dest=0)

        logging.debug('Sent exit signal to {}:{}'
                      .format(icomm, service_name))

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
