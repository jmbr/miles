"""Default values.

"""

__all__ = ['colors', 'default_randbits', 'global_convergence_tolerance', 'local_convergence_tolerance', 'marker_size', 'max_chunks', 'max_iterations', 'samples_per_milestone_per_iteration', 'save_interval', 'trajectory_max_trajectories']  # noqa: E501

import math

# Maximum number of chunks per trajectory fragment.
max_chunks = 1000000

# Number of transitions to gather between successive database back-ups.
save_interval = 1000

# Maximum number of transitions to sample when runing in
# long-trajectory mode.
trajectory_max_trajectories = int(100)

samples_per_milestone_per_iteration = int(1000)

max_iterations = 10000

# colors = 'YlGnBu'
colors = 'RdBu_r'
marker_size = 5

# When sampling from a particular milestone, the value of
# local_convergence_tolerance determines when the estimates are
# considered sufficiently good to proceed to the next iteration.
local_convergence_tolerance = 1e-3

global_convergence_tolerance = 1.0 / math.sqrt(max_iterations)

default_randbits = 128
