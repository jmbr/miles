"""Run iterative milestoning simulation."""


__all__ = ['run']

import logging
import math
import warnings
from typing import Optional

import numpy as np
import scipy.linalg as linalg

from miles import (Distributions, Iteration, Matrices, Milestones, Simulation, TimestepperServer, TransitionKernel, TransitionKernelError, bold, save_distributions)  # noqa: E501


def run(simulation: Simulation,
        initial_distributions: Distributions,
        num_iterations: int,
        num_samples: int,
        local_tolerance: float,
        global_tolerance: float,
        num_processes: int) -> None:
    """Orchestrates milestoning simulation.

    Parameters
    ----------
    simulation : Simulation
        Object containing the simulation parameters.
    initial_distributions : Distributions
        Initial distributions to use at the first iteration.
    num_iterations : int
        Maximum number of iterations to run.
    num_samples : int
        Minimum number of transitions to sample per milestone per
        iteration.
    local_tolerance : float
        Tolerance used for establishing convergence while sampling
        from each milestone.
    global_tolerance : float
        Tolerance used for establishing convergence of the iterative
        process.
    num_processes : int
        Number of MPI processes to use in parallel.

    """
    show_info(simulation.configuration, num_iterations, num_samples,
              global_tolerance, local_tolerance)

    milestones = simulation.milestones

    matrices = Matrices(milestones)
    kernel = TransitionKernel(matrices)

    ds = initial_distributions
    # q = make_initial_guess(milestones)

    timestepper_server = TimestepperServer(simulation, num_processes)
    try:
        timestepper_server.connect()

        for num_iteration in range(num_iterations):
            ds, q = run_iteration(num_iteration, simulation, kernel,
                                  ds, num_samples, local_tolerance,
                                  global_tolerance, timestepper_server)
            simulation.database.reload_zip_file()
    except StopIteration:
        pass
    except KeyboardInterrupt:
        logging.info('Interrupted. Waiting for jobs to end...')
        timestepper_server.stop()
    # except RuntimeError as exc:
    #     logging.error('Exiting due to run-time error: {}'.format(exc))
    finally:
        timestepper_server.disconnect()


def make_initial_guess(milestones: Milestones) -> np.array:
    """Return initial guess for the stationary flux vector.

    """
    M = milestones.max_milestones
    q = np.ones(M)
    return q / np.sum(q)


def show_info(configuration, num_iterations, num_samples,
              global_tolerance, local_tolerance):
    """Display information about the simulation.

    """
    for line in str(configuration).split('\n'):
        logging.info(line)

    logging.info('Running at most {} iterations until global '
                 'convergence is attained with a tolerance of {:g}.'
                 .format(num_iterations, global_tolerance))

    logging.info('Drawing at least {} samples per milestone per '
                 'iteration until local convergence is attained with '
                 'a tolerance of {:g}. '
                 .format(num_samples, local_tolerance))


def run_iteration(num_iter: int, simulation: Simulation,
                  global_kernel: TransitionKernel,
                  initial_distributions: Distributions, min_samples: int,
                  local_tolerance: float, global_tolerance: float,
                  server: TimestepperServer,
                  prev_q: Optional[np.array] = None):
    """Run a full iteration, report results, and prepare for the next.

    """
    logging.info(bold('Iteration {}'.format(num_iter)))

    if len(initial_distributions) == 0:
        logging.error('No first hitting points available.')
        raise RuntimeError

    iteration = Iteration(simulation, global_kernel,
                          initial_distributions, min_samples,
                          local_tolerance, num_iter, server)

    statdist, statvec = iteration.run(prev_q)

    try:
        statdist_global, statvec_global = global_kernel.compute_distributions()
    except TransitionKernelError as exc:
        warnings.warn('Unable to compute eigenvector: {}'.format(exc))

    save_results(iteration, statdist, statdist_global)

    global_matrices = global_kernel.matrices
    local_matrices = iteration.transition_kernel.matrices

    # Compute initial distributions for the next iteration.
    # new_initial_distributions = statdist
    # new_q = global_matrices.q
    new_initial_distributions = statdist
    new_q = statvec

    logging.info('Mean first passage time: {:4f} units of '
                 'time [current].'.format(local_matrices.mfpt))
    logging.info('Mean first passage time: {:4f} units of '
                 'time [overall].'.format(global_matrices.mfpt))

    if has_converged(local_matrices.q, prev_q, global_tolerance):
        logging.info('The iterative process converged after '
                     '{} iterations.'.format(num_iter))
        raise StopIteration

    return new_initial_distributions, new_q


def save_results(iteration: Iteration,
                 initial_distributions: Distributions,
                 stationary_distributions: Distributions) -> None:
    """Save database, matrices, and new initial distributions.

    """
    def save_kernel(n, kernel, tag=''):
        kernel.save_distributions(suffix='-{:04d}'.format(n))

        matrices = kernel.matrices
        matrices.save('K-{}{:04d}.mtx'.format(tag, n),
                      'T-{}{:04d}.mtx'.format(tag, n),
                      'q-{}{:04d}.dat'.format(tag, n),
                      't-{}{:04d}.dat'.format(tag, n),
                      'p-{}{:04d}.dat'.format(tag, n))

    n = iteration.number

    save_kernel(n, iteration.transition_kernel)
    save_distributions(initial_distributions,
                       'initial-{:04d}.dst'.format(n+1))

    save_kernel(n, iteration.global_transition_kernel, 'global-')
    save_distributions(stationary_distributions,
                       'initial-global-{:04d}.dst'.format(n+1))


def has_converged(v: np.array, prev_v: np.array, tolerance: float) \
        -> bool:
    """Determine whether the convergence criterion is satisfied.

    Parameters
    ----------
    v : np.array
        Estimated stationary flux vector.
    prev_v : np.array
        Stationary flux vector from previous iteration.
    tolerance : float
        Threshold used for convergence criterion.

    Returns
    -------
    is_dominant : bool
        True, if the vector v is the dominant (left-)eigenvector of
        the matrix K. False, otherwise

    """
    if prev_v is None:
        return False

    norm = linalg.norm(v - prev_v, np.inf) / linalg.norm(v, np.inf)
    logging.debug('Relative error between previous and current '
                  'stationary vectors: {:.3g}.'.format(norm))

    is_dominant = math.isclose(norm, 0.0, rel_tol=tolerance)

    return is_dominant
