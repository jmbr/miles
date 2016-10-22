"""Parse configuration files from the colvars program.

This module is intended so that miles can understand input files from
colvars. That way, the user only has to be concerned with writing a
single configuration file specifying the collective variables.

This is not a full parser of the colvars input syntax. It understands
the colvars format enough to function but the user is resposible for
verifying that the collective variables read in this module agree with
those in colvars.

"""

__all__ = ['ColvarsParserError', 'ColvarsParser']


import io
import math
from typing import Optional, Union, Tuple

from miles import CollectiveVariable, CollectiveVariables, Interval, PeriodicInterval  # noqa: E501


OFF_VALUES = {'no', 'off', 'false'}


class ColvarsParserError(Exception):
    pass


class Token:
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class OpenBracket(Token):
    pass


class CloseBracket(Token):
    pass


class Symbol(Token):
    def __init__(self, symbol):
        self.symbol = symbol

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.symbol)

    def __eq__(self, other):
        """Case-insensitive equality predictate.

        """
        if not isinstance(other, self.__class__):
            return False
        return self.symbol.lower() == other.symbol.lower()


class ColvarsParser:
    """Interpreted colvars input files.

    """
    indices = []                # type: List[int]
    collective_variables = []   # type: List[CollectiveVariable]

    def __init__(self, arg: Union[str, io.TextIOBase]) -> None:
        self.parse(arg)

    @classmethod
    def _lexer(cls, colvars_file):
        for line in colvars_file.readlines():
            for item in line.strip().split():
                if item.startswith('#'):
                    break
                elif item == '{':
                    yield OpenBracket()
                elif item == '}':
                    yield CloseBracket()
                else:
                    yield Symbol(item)

    @classmethod
    def _parser(cls, colvars_file):
        colvars = []
        current_context = colvars

        for token in cls._lexer(colvars_file):
            if isinstance(token, OpenBracket):
                colvars.append(current_context)
                current_context = []
            elif isinstance(token, Symbol):
                current_context.append(token)
            elif isinstance(token, CloseBracket):
                prev_context = colvars.pop()
                prev_context.append(current_context)
                current_context = prev_context

        return colvars

    @classmethod
    def _find(cls, colvar, symbol_name: str) -> Tuple[bool, Optional[str]]:
        """Find a symbol in a parsed colvars file.

        Returns
        -------
        found : bool
            Whether the symbol was found or not.
        value : Optional[str]
            Accompanying value to the symbol if found, `None` otherwise.

        """
        for i, c in enumerate(colvar):
            if c == Symbol(symbol_name):
                if i+1 < len(colvar) and isinstance(colvar[i+1], Symbol):
                    return True, colvar[i+1].symbol
                else:
                    return True, None

        return False, None

    @classmethod
    def _is_periodic(cls, colvar):
        """Determine if a collective variable is periodic.

        """
        is_periodic = False
        periodic_keywords = ['dihedral', 'spinAngle']  # distanceZ
        for keyword in periodic_keywords:
            found, value = cls._find(colvar, keyword)
            if found:
                is_periodic = True
                break
        return is_periodic

    @classmethod
    def _get_boundaries(cls, colvar) -> Tuple[float, float]:
        """Obtain the boundaries of the collective variable.

        Returns
        -------
        ab : Tuple[float, float]
            Endpoints of the interval where the collective variable is
            defined.

        """
        a_found, a_str = cls._find(colvar, 'lowerBoundary')
        if a_found:
            a = float(a_str)
        else:
            a = -math.inf

        b_found, b_str = cls._find(colvar, 'upperBoundary')
        if b_found:
            b = float(b_str)
        else:
            b = math.inf

        return a, b

    @classmethod
    def _is_enabled(cls, colvar, keyword: str) -> bool:
        """Determine if a keyword has been explicitly enabled.

        Notes
        -----
            It is in general not true that::
                not _is_enabled(cv, kw) == _is_disabled(cv, kw)

        """
        found, value = cls._find(colvar, keyword)

        if isinstance(value, Symbol):
            return found and (not value or value not in OFF_VALUES)
        else:
            return found

    @classmethod
    def _is_disabled(cls, colvar, keyword: str) -> bool:
        """Determine if a keyword has been explicitly disabled.

        See also
        --------
        _is_enabled

        """
        found, value = cls._find(colvar, keyword)

        if found:
            return value.lower() in OFF_VALUES
        else:
            return False

    def parse(self, arg: Union[str, io.TextIOBase]) -> None:
        """Interpret collective variables in colvars configuration file.

        Parameters
        ----------
        arg : Union[str, file]
            The input configuration. This can be either a file name or
            an open file.

        Raises
        ------
        ColvarsParserError

        """
        if isinstance(arg, str):
            with open(arg) as f:
                colvars = self._parser(f)
        else:
            colvars = self._parser(arg)

        all_colvars = [colvars[i+1] for i, c in enumerate(colvars)
                       if c == Symbol('colvar')]

        prev_index = 1
        index = prev_index

        cvs = []
        indices = []

        for colvar in all_colvars:
            found, name = self._find(colvar, 'name')
            if not found:
                raise ColvarsParserError('Collective variable with no name')

            a, b = self._get_boundaries(colvar)

            if self._is_periodic(colvar):
                interval = PeriodicInterval()
            else:
                interval = Interval(a, b)

            for keyword in {'outputVelocity', 'outputEnergy',
                            'outputSystemForce', 'outputAppliedForce'}:
                if self._is_enabled(colvar, keyword):
                    index += 1

            if not self._is_disabled(colvar, 'outputValue'):
                index += 1
                cv = CollectiveVariable(name, interval)
                indices.append(prev_index)
                cvs.append(cv)

            prev_index = index

        self.collective_variables = CollectiveVariables(cvs, indices)

    def __str__(self) -> str:
        s = '=' * 77 + '\n'

        fields = ['Column', 'Name', 'Lower bound', 'Upper bound', 'Periodic?']
        s += '{:>6} | {:<14} | {:<12} | {:<12} | {}\n'.format(*fields)

        s += '-' * 77 + '\n'

        fmt = '{:>6} | {:<14} | {:<12} | {:<12} | {}\n'

        cvs = self.collective_variables
        for idx, cv in zip(cvs.indices, cvs):
            name = cv.name
            a, b = cv.codomain.a, cv.codomain.b
            periodic = cv.codomain.periodic
            s += fmt.format(idx, name, a, b, periodic)

        s += '=' * 77

        return s
