"""Management of distributions of items.

Examples
--------
First, let's create a few empirical distributions::
>>> d1 = Distribution([1, 2, 3])
>>> d2 = DeltaDistribution(23)
>>> d3 = Distribution([10, 11, 12])
>>> d4 = DeltaDistribution(42)
>>> d1
Distribution([1, 2, 3])
>>> d2
DeltaDistribution(23)
>>> d3
Distribution([10, 11, 12])
>>> d4
DeltaDistribution(42)

Next, we obtain various convex combinations of them::
>>> a = 1/3*d1 + 2/3*d2
>>> b = 1/2*d1 + 1/2*d2
>>> c = 1/4*d1 + 1/4*d2 + 1/2*d3
>>> d  = 1/4*d1 + 1/4*d2 + 1/4*d3 + 1/4*d4
>>> a
SumDistribution([WeightedDistribution(0.3333333333333333, Distribution([1, 2, 3])), WeightedDistribution(0.6666666666666666, DeltaDistribution(23))])
>>> b
SumDistribution([WeightedDistribution(0.5, Distribution([1, 2, 3])), WeightedDistribution(0.5, DeltaDistribution(23))])
>>> c
SumDistribution([WeightedDistribution(0.25, Distribution([1, 2, 3])), WeightedDistribution(0.25, DeltaDistribution(23)), WeightedDistribution(0.5, Distribution([10, 11, 12]))])
>>> d
SumDistribution([WeightedDistribution(0.25, Distribution([1, 2, 3])), WeightedDistribution(0.25, DeltaDistribution(23)), WeightedDistribution(0.25, Distribution([10, 11, 12])), WeightedDistribution(0.25, DeltaDistribution(42))])

"""

__all__ = ['BaseDistribution', 'DeltaDistribution', 'Distribution', 'DistributionError', 'SumDistribution', 'WeightedDistribution', 'compute_histogram']  # noqa: E501

import math
from abc import ABCMeta, abstractmethod
from collections import Sized
from typing import List, Optional

import numpy as np


MIN_WEIGHT = 0.0
MAX_WEIGHT = 1.0


class DistributionError(Exception):
    """Distribution-related error."""
    pass


class BaseDistribution(Sized, metaclass=ABCMeta):
    """Probability distribution.

    """
    @abstractmethod
    def sample(self) -> object:
        """Draw a sample with replacement."""
        raise NotImplementedError('Method must be implemented by subclass')

    def is_zero(self) -> bool:
        """Determine if an empirical distribution is empty."""
        return len(self) == 0

    @abstractmethod
    def __len__(self) -> int:
        """Number of data points available."""
        raise NotImplementedError('Method must be implemented by subclass')

    @abstractmethod
    def get_items(self):
        raise NotImplementedError('Method must be implemented by subclass')

    def __rmul__(self, weight: float) -> 'WeightedDistribution':
        """Scalar multiplication.

        This method wraps any distribution inside a weighted
        distribution.

        Parameters
        ----------
        weight : float

        Raises
        ------
        DistributionError
            In case the weight is invalid.

        """
        if (not isinstance(weight, float)
                or not (MIN_WEIGHT <= weight <= MAX_WEIGHT)):
            raise DistributionError('Invalid weight value', weight)
        return WeightedDistribution(weight, self)

    def __add__(self, other: 'BaseDistribution') -> 'BaseDistribution':
        """Convex combination of two distributions.

        """
        if other.is_zero():
            return self
        if self.is_zero() and isinstance(other, BaseDistribution):
            return other
        else:
            return NotImplemented


class WeightedDistribution(BaseDistribution):
    """Weighted empirical distributions.

    Attributes
    ----------
    weight : float
        A weight in the unit interval for the distribution.
    distribution : BaseDistribution
        An instance of BaseDistribution implementing the sample
        method, etc.

    """
    def __init__(self, weight: float,
                 distribution: BaseDistribution) -> None:
        if not (MIN_WEIGHT <= weight <= MAX_WEIGHT):
            raise ValueError('Invalid weight value: {}'.format(weight))
        self.weight = weight
        self.distribution = distribution

    def sample(self) -> object:
        return self.distribution.sample()

    def is_zero(self) -> bool:
        """Determine if an empirical distribution is empty."""
        return self.weight == 0.0 or super().is_zero()

    def __len__(self) -> int:
        return len(self.distribution)

    def __repr__(self) -> str:
        return ('{}({!r}, {!r})'
                .format(self.__class__.__name__,
                        self.weight,
                        self.distribution))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WeightedDistribution):
            return (self.weight == other.weight
                    and self.distribution == other.distribution)
        else:
            return False

    def __add__(self, other: 'BaseDistribution') -> BaseDistribution:
        """Addition of weighted distributions.

        """
        if other.is_zero():
            return self
        elif self.is_zero() and isinstance(other, BaseDistribution):
            return other
        elif isinstance(other, WeightedDistribution):
            return SumDistribution([self, other])
        else:
            return NotImplemented

    def get_items(self):
        for item in self.distribution.get_items():
            yield item


class SumDistribution(BaseDistribution):
    """Convex combination of distributions.

    Attributes
    ----------
    weights : np.array
        Array of weights for the convex combination.
    probabilities : np.array
        Normalized array of weights suitable for use as a discrete
        probability distribution.
    distributions : List[WeightedDistribution]
        Sequence of distributions. There must be as many distributions
        as weights.

    """
    def __init__(self, distributions: List[WeightedDistribution]) -> None:
        if not all(isinstance(d, WeightedDistribution) for d in distributions):
            raise DistributionError('Invalid distributions.', distributions)

        self.weights = np.array([d.weight for d in distributions])
        total = sum(self.weights)

        # Sanity-check weights.
        if not all(weight >= 0.0 for weight in self.weights):
            raise DistributionError('Negative weight values found.')
        if total > 1.0:
            raise DistributionError('Invalid weights: their sum is {} > 1.'
                                    .format(total))

        # Store normalized weights for sampling.
        self.probabilities = self.weights / total

        self.distributions = distributions

    def sample(self):
        assert math.isclose(sum(self.probabilities), 1.0)

        distribution = np.random.choice(self.distributions,
                                        p=self.probabilities)
        return distribution.sample()

    def __len__(self):
        return sum(len(d) for d in self.distributions)

    def __repr__(self) -> str:
        return '{}({!r})'.format(self.__class__.__name__,
                                 self.distributions)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SumDistribution):
            return (self.weights == other.weights
                    and self.distributions == other.distributions)
        else:
            return NotImplemented

    def __add__(self, other: BaseDistribution) -> 'SumDistribution':
        """Addition of sums of weighted distributions."""
        if isinstance(other, WeightedDistribution):
            return SumDistribution(self.distributions + [other])
        else:
            return NotImplemented

    def get_items(self):
        for distribution in self.distributions:
            for item in distribution.get_items():
                yield item


class Distribution(BaseDistribution):
    """Empirical distribution.

    This class contains data points that can be added to it and
    sampled afterwards.

    Attributes
    ----------
    _items : Optional[List[object]]
        List of items to be sampled.

    """
    def __init__(self, items: Optional[List[object]] = None) -> None:
        if items is None:
            self._items = []    # type: Optional[List[object]]
        else:
            self._items = items

    def sample(self) -> object:
        """Draw a random sample with replacement."""
        # Invoking numpy.random.choice on a range of integers is much
        # faster than passing self._items directly.
        idx = np.random.choice(len(self._items))
        return self._items[idx]

    def update(self, item: object) -> None:
        """Add a new data point to the empirical distribution.

        """
        self._items.append(item)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return '{}({!r})'.format(self.__class__.__name__,
                                 self._items)

    def get_items(self):
        for item in self._items:
            yield item


class DeltaDistribution(Distribution):
    """Delta distribution concentrated at a single point.

    """
    def __init__(self, item: object) -> None:
        super().__init__([item])

    def update(self, item: object) -> None:
        msg = 'DeltaDistribution does not provide an update method'
        raise NotImplementedError(msg)

    def __repr__(self) -> str:
        return '{}({!r})'.format(self.__class__.__name__,
                                 self._items[0])


def compute_histogram(distribution, total_samples, normalize=True):
    """Compute histogram by sampling from a distribution.

    Parameters
    ----------
    distribution : Distribution
        An empirical distribution to sample from.
    total_samples : int
        Number of samples to draw.
    normalize : bool
        Whether the returned histogram should be normalized or not.

    Returns
    -------
    histogram : dict
        Histogram of the frequency of each particular sample. Possibly
        normalized.

    """
    histogram = {}

    for n in range(total_samples):
        sample = distribution.sample()
        count = histogram.get(sample, 0) + 1
        histogram[sample] = count

    if normalize:
        return {k: v / total_samples for k, v in histogram.items()}
    else:
        return histogram
