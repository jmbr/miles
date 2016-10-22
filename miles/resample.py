__all__ = ['resample']

import itertools
import logging

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        """Dummy replacement for progress bar."""
        x

from miles import (Database, Distribution, Matrices, TransitionKernel, bold, save_distributions)  # noqa: E501


def resample(simulation, samples, database_dir,
             stationary_distributions, transition_matrix,
             lag_time_matrix, stationary_flux, local_mfpts,
             stationary_probability):
    logging.info('Running resampling...')

    milestones = simulation.milestones

    reactant_distribution = simulation.reactant_distribution

    database = simulation.database
    kernel = TransitionKernel(database)

    kernel.compute_distributions()
    K = kernel.matrices.K

    new_database = Database(database.anchors)
    new_matrices = Matrices(simulation.milestones)
    new_kernel = TransitionKernel(new_matrices)

    tuples = []
    for final in kernel.keys():
        for initial in kernel[final].keys():
            i, j = initial.index, final.index
            dist = kernel[final][initial]
            tuples.append((i, j, K[i, j], dist))

    sorted_tuples = sorted(tuples, key=lambda x: x[0])
    grouped_tuples = itertools.groupby(sorted_tuples, lambda x: x[0])

    for initial, grouped in grouped_tuples:
        distribution = Distribution()
        for t in list(grouped):
            distribution += t[2] * t[3]

        logging.info('Sampling from Milestone {}.'.format(initial))

        # Resample kernel obtained from database and copy the results
        # into new_kernel.
        for _ in tqdm(range(samples)):
            transition = distribution.sample()
            new_kernel.update(transition)
            new_database.update(transition)
            if transition.final_milestone in milestones.products:
                new_transition = reactant_distribution.sample()
                new_transition.lag_time = 0.0
                new_transition.initial_milestone = transition.final_milestone
                new_database.update(new_transition)
                new_kernel.update(new_transition)

    new_distributions = new_kernel.compute_distributions()
    save_distributions(new_distributions, stationary_distributions)

    new_kernel.matrices.save(transition_matrix, lag_time_matrix,
                             stationary_flux, local_mfpts,
                             stationary_probability)
    mfpt = new_kernel.matrices.mfpt
    logging.info(bold('Mean first passage time: {} units of '
                      'time.'.format(mfpt)))
    new_database.save(database_dir)

    logging.info('Done.')
