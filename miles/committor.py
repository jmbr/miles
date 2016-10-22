"""Compute committor function on milestones.

"""

__all__ = ['committor']


from typing import Set

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def committor(K: scipy.sparse.csr_matrix,
              reactants: Set[int], products: Set[int]) -> np.array:
    """Compute the committor function on the milestones.

    Parameters
    ----------
    K : scipy.sparse.csr_matrix
        Transition matrix coming from an equilibrium simulation (i.e.,
        a simulation in which no boundary conditions have been
        imposed).
    reactants : Set[int]
        Set of indices identifying the milestones that belong to the
        reactant state.
    products : Set[int]
        Set of indices identifying the milestones that belong to the
        product state.

    Returns
    -------
    committor_vector : np.array
        Vector of values of the committor function at the milestone
        corresponding to each index.

    """
    assert K.shape[0] == K.shape[1]
    M = K.shape[0]

    # Construct auxiliary matrix and solve for the values of the
    # committor function.
    reactant_indices = set(reactants)
    product_indices = set(products)
    reactant_and_product_indices = reactant_indices.union(product_indices)

    # Convert set of valid indices to list to impose ordering.
    valid_indices = set(range(M)) - reactant_and_product_indices
    index_mapping = {i: k for k, i in enumerate(valid_indices)}

    N = len(valid_indices)

    A = scipy.sparse.lil_matrix((N, N))
    b = np.zeros(N)
    for i in valid_indices:
        ii = index_mapping[i]
        A[ii, ii] = 1

        for j in product_indices:
            b[ii] += K[i, j]

        _, nzj = K[i, :].nonzero()
        if len(nzj) == 0:
            continue

        for j in valid_indices.intersection(set(nzj)):
            jj = index_mapping[j]
            A[ii, jj] -= K[i, j]

    c = scipy.sparse.linalg.spsolve(A.tocsr(), b)

    # Convert the solution into a vector whose entries are the values
    # of the committor function for the corresponding milestone.
    valid = np.fromiter((i in valid_indices for i in range(M)),
                        dtype=np.bool)

    committor_vector = np.zeros(M)
    committor_vector[valid] = c
    products_array = np.fromiter(products, dtype=np.int)
    committor_vector[products_array] = 1.0

    return committor_vector
