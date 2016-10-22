"""Manage finite-dimensional transition and time matrices throughout a
simulation.

"""


__all__ = ['Matrices', 'make_stochastic_matrix', 'make_time_matrix']

import logging
import sys
from typing import Optional

import scipy.io
import scipy.sparse as sparse
import numpy as np

from miles import (Milestones, Transition, dominant_eigenvector)


class Matrices:
    """Finite-dimensional operators estimated during simulations.

    Each entry in the matrix A counts the number of transitions
    between the corresponding milestones. Each entry in the matrix B
    contains the aggregated lag times for the corresponding
    transitions.

    """

    A = None                 # type: Optional[scipy.sparse.dok_matrix]
    B = None                 # type: Optional[scipy.sparse.dok_matrix]
    _K = None                # type: Optional[scipy.sparse.csr_matrix]
    _T = None                # type: Optional[scipy.sparse.csr_matrix]
    _q = None                # type: Optional[np.array]
    _t = None                # type: Optional[np.array]
    _p = None                # type: Optional[np.array]
    _eigval = None           # type: Optional[float]
    _mfpt = None             # type: Optional[float]

    def __init__(self, milestones: Milestones) -> None:
        self.milestones = milestones

        num_milestones = milestones.max_milestones
        M = num_milestones
        self.A = sparse.dok_matrix((M, M), dtype=np.int32)
        self.B = sparse.dok_matrix((M, M), dtype=np.float64)

        self.product_indices = [p.index for p in milestones.products]

        self._connect_products_to_reactants(milestones)

        self._reset_computed_quantities()

    def _reset_computed_quantities(self) -> None:
        """Reset computed quantities.

        Set computed quantities to None so that they are explicitly
        recomputed in the future when they are needed.

        """
        self._K = None
        self._T = None
        self._q = None
        self._t = None
        self._p = None
        self._eigval = None
        self._mfpt = None

    # XXX Think about eliminating these property methods.
    @property
    def K(self) -> np.array:
        """Transition matrix."""
        if self._K is None:
            self._K = make_stochastic_matrix(self.A)
        return self._K

    @property
    def T(self) -> np.array:
        """Lag time matrix."""
        if self._T is None:
            self._T = make_time_matrix(self.A, self.B)
        return self._T

    @property
    def q(self) -> np.array:
        """Stationary flux vector."""
        # if self._q is None:
        #     self.compute()
        assert self._q is not None
        return self._q

    @property
    def t(self) -> np.array:
        """Vector of local mean first passage times."""
        # if self._t is None:
        #     self.compute()
        assert self._t is not None
        return self._t

    @property
    def p(self) -> np.array:
        """Vector of stationary probabilities."""
        # if self._p is None:
        #     self.compute()
        assert self._p is not None
        return self._p

    @property
    def eigval(self) -> float:
        """Dominant eigenvalue (should be one or very close to it)."""
        # if self._eigval is None:
        #     self.compute()
        return self._eigval

    @property
    def mfpt(self) -> float:
        """Global mean first passage time."""
        # if self._mfpt is None:
        #     self.compute()
        assert self._mfpt is not None
        return self._mfpt

    def compute(self, prev_q: Optional[np.array] = None) -> float:
        """Compute observables.

        Obtain the stationary flux, local mean first passage times,
        global mean first passage time, and stationary probability.

        Parameters
        ----------
        prev_q : np.array, optional
            A guess for the stationary flux vector

        Returns
        -------
        mfpt : float
            The value of the mean first passage time if the
            computation was successful, None otherwise.

        Raises
        ------
        ValueError
            If something goes wrong with the eigenvalue computation.

        """
        self._K = make_stochastic_matrix(self.A)
        self._T = make_time_matrix(self.A, self.B)
        K, T = self._K, self._T

        if prev_q is not None:
            logging.debug('Computing matrix-vector product.')
            self._eigval = None
            self._q = K.T.dot(prev_q)
            self._q /= np.sum(self._q)  # Normalize (K could be substochastic)
        else:
            # The dominant_eigenvector method may raise an exception if
            # the transition matrix is not valid.  We simply let the
            # exception propagate.
            logging.debug('Solving eigenvalue problem.')
            self._eigval, self._q = dominant_eigenvector(K.T)

        # Compute vector of local MFPTs.
        M = self.A.shape[0]
        e = np.ones(M, dtype=np.float64)
        KT = K.multiply(T).toarray()
        self._t = KT.dot(e)

        # Compute vector of stationary probabilities.
        p = np.multiply(self.q, self.t)
        self._p = p / np.linalg.norm(p, 1)

        # Compute MFPT.
        if self.product_indices:
            sum_products = self.q[self.product_indices].sum()
            self._mfpt = self.q.dot(self.t) / sum_products

        return self._mfpt

    def update(self, transition: Transition) -> None:
        """Update matrices with a transition.

        Transitions are assumed to be from a trajectory fragment
        started and stopped at two different milestones.  No
        consistency checks are done in this method.

        """
        initial = transition.initial_milestone
        if initial not in self.milestones.products:
            final = transition.final_milestone
            i, j = initial.index, final.index
            self.A[i, j] += 1
            self.B[i, j] += transition.lag_time
            self._reset_computed_quantities()

    def save(self, transition_matrix: str, lag_time_matrix: str,
             stationary_flux: str, local_mfpts: str,
             stationary_probability: str) -> None:
        """Save matrices to files."""
        scipy.io.mmwrite('A.mtx', self.A)
        scipy.io.mmwrite('B.mtx', self.B)

        logging.debug('Saving transition matrix to {!r}.'
                      .format(transition_matrix))
        scipy.io.mmwrite(transition_matrix, self.K)

        logging.debug('Saving lag time matrix to {!r}.'
                      .format(lag_time_matrix))
        scipy.io.mmwrite(lag_time_matrix, self.T)

        logging.debug('Saving stationary flux vector to {!r}.'
                      .format(stationary_flux))
        np.savetxt(stationary_flux, self.q)

        logging.debug('Saving vector of local MFPTs to {!r}.'
                      .format(local_mfpts))
        np.savetxt(local_mfpts, self.t)

        logging.debug('Saving vector of stationary probabilities {!r}.'
                      .format(stationary_probability))
        np.savetxt(stationary_probability, np.multiply(self.q, self.t))

    def _connect_products_to_reactants(self, milestones: Milestones) -> None:
        """Modify matrices so that the product goes to the reactant."""

        if not milestones.products or not milestones.reactants:
            return

        # XXX Add support for multiple reactants and products.
        i = milestones.products[0].index
        j = milestones.reactants[0].index

        A = self.A
        B = self.B

        # If we come across the result from an equilibrium simulation,
        # where the product milestone is sending trajectories to other
        # neighbors (instead of to the product), then we zero out all
        # the entries from the product to its neighbors before
        # connecting it to the reactant.
        _, k = A[i, :].nonzero()
        A[i, k] = 0

        # Now connect the product to the reactant.
        A[i, j] = 1

        # We add a very small value to B.  Setting B[j, i] to zero
        # would not work because these are sparse matrices and zeros
        # are not stored.
        B[i, j] = sys.float_info.epsilon


def make_stochastic_matrix(A: np.array) -> np.array:
    """Convert to row-stochastic matrix.

    Normalizes (in the 1-norm) each row of the matrix A and returns
    the result.
    """
    K = sparse.csr_matrix(A, dtype=np.float64)

    for i in range(K.shape[0]):
        l = K.indptr[i:i + 2]
        if l[0] == l[1]:
            continue
        data = K.data[l[0]:l[1]]
        norm1 = np.sum(data)
        assert norm1 > 0.0
        data = data / norm1
        K.data[l[0]:l[1]] = data

    return K


def make_time_matrix(A: np.array, B: np.array) -> np.array:
    """Compute matrix of average local lag times."""
    assert A.shape == B.shape

    AA = sparse.csr_matrix(A, dtype=np.float64)
    T = sparse.csr_matrix(B, dtype=np.float64)

    for i in range(T.shape[0]):
        lt = T.indptr[i:i+2]
        if lt[0] == lt[1]:
            continue

        la = AA.indptr[i:i+2]

        Tdata = T.data[lt[0]:lt[1]]
        AAdata = AA.data[la[0]:la[1]]
        data = np.divide(Tdata, AAdata)
        T.data[lt[0]:lt[1]] = data

    return T
