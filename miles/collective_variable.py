"""Collective variable.

A collective variable is a mapping that takes a phase space point and
assigns to it a value in a certain codomain. This codomain is
typically a closed interval, the real line (e.g., distance between
atoms), or the unit circle (e.g., dihedral angles).

"""


__all__ = ['Codomain', 'CollectiveVariable', 'CollectiveVariableError', 'Interval', 'PeriodicInterval']  # noqa: E501


import math
from abc import ABCMeta, abstractmethod
from typing import Callable, Optional

from miles import PhaseSpacePoint


class Codomain(metaclass=ABCMeta):
    """Codomain of a collective variable.

    This class represents an interval on the real line (or possibly
    the real line with PBC).

    Attributes
    ----------
    a : float
        Left end-point of the interval
    b : float
        Right end-point of the interval
    periodic : bool
        True if the interval is periodic, false otherwise.

    """

    def __init__(self, a: float, b: float, periodic: bool) -> None:
        self.a = a
        self.b = b
        self.periodic = periodic

    def __repr__(self):
        return ('{}({}, {}, periodic={})'
                .format(self.__class__.__name__, self.a, self.b,
                        self.periodic))

    @abstractmethod
    def distance(self, x: float, y: float) -> float:
        """Compute the distance between two reaction coordinates."""
        raise NotImplementedError


class Interval(Codomain):
    def __init__(self, a: float, b: float) -> None:
        if a is None:
            a = -math.inf
        if b is None:
            b = math.inf
        super().__init__(a, b, False)

    def distance(self, x: float, y: float) -> float:
        assert self.a <= x <= self.b
        assert self.a <= y <= self.b
        return abs(x - y)


class PeriodicInterval(Codomain):
    """Unit circle."""
    def __init__(self, a: Optional[float] = -180, b: Optional[float] = 180) \
            -> None:
        super().__init__(a, b, True)
        self.period = self.b - self.a

    def distance(self, x: float, y: float) -> float:
        a, b, p = self.a, self.b, self.period
        return min(abs(a - b - p),
                   abs(a - b),
                   abs(a - b + p))


class CollectiveVariableError(Exception):
    """Error related to a collective variable.

    """
    pass


class CollectiveVariable:
    """Collective variable."""

    def __init__(self, name: str, codomain: Codomain,
                 function: Optional[Callable[[PhaseSpacePoint], float]]=None) \
            -> None:
        self.name = name        # type: str
        self.codomain = codomain  # type: Codomain
        self.function = function

    def __call__(self, p: PhaseSpacePoint) -> float:
        """Map a phase space point into a reaction coordinate.

        Parameters
        ----------
        p : PhaseSpacePoint
            A phase space point.

        Returns
        -------
        colvar : float
            Value of the reaction coordinate (collective variable) at
            the point `p`.

        Raises
        ------
        CollectiveVariableError
            In case there is no function available to compute the
            collective variable.

        """
        if self.function:
            return self.function.__call__(p)
        else:
            raise CollectiveVariableError()

    def __repr__(self) -> str:
        return ('{}({!r}, {!r}, function={!r})'
                .format(self.__class__.__name__,
                        self.name,
                        self.codomain,
                        self.function))
