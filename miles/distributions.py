"""Management of collections of empirical distributions.

"""

__all__ = ['Distributions', 'WeightedDistributions', 'load_distributions', 'save_distributions']  # noqa: E501

import itertools
import numbers
import pickle


class Distributions(dict):
    """Dictionary data structure mapping keys to empirical distributions.

    """
    def __init__(self, *args, **kwargs):
        super(Distributions, self).__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return '{}({!r})'.format(self.__class__.__name__,
                                 super(Distributions, self))

    def __rmul__(self, weight):
        if not isinstance(weight, numbers.Real):
            raise ValueError
        else:
            return WeightedDistributions(weight, self)


class WeightedDistributions:
    """Weighted collection of distributions on milestones.

    Notes
    -----
    This class does not inherit from Distributions.

    """
    def __init__(self, weight, distributions):
        self.weight = weight
        self.distributions = distributions

    def __repr__(self):
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.weight,
                                       self.distributions)

    def __add__(self, other):
        """Return sum of weighted distributions.

        """
        if not isinstance(other, WeightedDistributions):
            return NotImplemented

        self_distributions = self.distributions
        other_distributions = other.distributions

        w1 = self.weight
        w2 = other.weight

        new_dist = Distributions()

        all_milestones = itertools.chain(self_distributions.keys(),
                                         other_distributions.keys())

        for milestone in all_milestones:
            d1 = self_distributions.get(milestone)
            d2 = other_distributions.get(milestone)

            if d1 is not None and d2 is not None:
                new_dist[milestone] = w1 * d1 + w2 * d2
            else:
                if d1 is not None:
                    new_dist[milestone] = d1
                else:
                    new_dist[milestone] = d2

        return new_dist


def load_distributions(file_name: str) -> Distributions:
    """Load a set of distributions from a file."""
    with open(file_name, 'rb') as f:
        ds = pickle.load(f)

    return ds


def save_distributions(distributions: Distributions, file_name: str) -> None:
    """Save a set of distributions to a file."""
    with open(file_name, 'wb') as f:
        pickle.dump(distributions, f)
