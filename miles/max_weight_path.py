"""Compute maximum weight path.

"""

__all__ = ['max_weight_path']


import logging
from typing import List, Tuple

import numpy as np
import scipy.sparse
import networkx


# Meta-reactant and meta-product
META_REACTANT = -np.inf
META_PRODUCT = np.inf


def make_connectivity_graph(transition_matrix: scipy.sparse.coo_matrix,
                            stationary_vector: np.array,
                            reactants: np.array, products: np.array) \
        -> networkx.MultiDiGraph:
    """Create flux-based graph.

    Parameters
    ----------
    transition_matrix : scipy.sparse.coo_matrix
        Transition matrix for the system.
    stationary_vector : np.array
        Vector of stationary flux weights.
    reactants : np.array
        Array of indices corresponding to the reactant milestones.
    products : np.array
        Array of indices corresponding to the products milestones.

    Returns
    -------
    G : networkx.MultiDiGraph
        Directed graph with the edges between milestones weighted
        according to the stationary flux vector.

    """
    G = networkx.MultiDiGraph()

    # Add meta nodes to the graph.
    for j in reactants:
        G.add_edge(META_REACTANT, j+1, weight=stationary_vector[j])
    for i in products:
        G.add_edge(i+1, META_PRODUCT, weight=0)

    num_milestones = transition_matrix.shape[0]
    assert num_milestones == stationary_vector.shape[0]

    for i, j in zip(transition_matrix.row, transition_matrix.col):
        if i in products:
            continue
        G.add_edge(i+1, j+1, weight=stationary_vector[j])

    return G


def compute_max_weight(G: networkx.MultiDiGraph,
                       start: int = META_REACTANT) -> None:
    """Compute maximum weight paths.

    This function annotates each node of the graph with its maximum
    weight from the starting node and a link to its predecessor.

    """
    def extract_max(labels):
        max_weight = -1
        max_label = None
        for label in labels:
            if G.node[label]['max_weight'] > max_weight:
                max_weight = G.node[label]['max_weight']
                max_label = label

        if max_label is not None:
            labels.remove(max_label)

        return max_label, max_weight

    labels = []
    for label in G.nodes_iter():
        G.node[label]['max_weight'] = np.inf if label == start else -1
        G.node[label]['predecessor'] = None
        G.node[label]['bottleneck'] = None
        labels.append(label)

    while labels:
        # Find out which node has maximal weight.
        current, weight = extract_max(labels)

        for neighbor in G[current]:
            m = min(weight, G.edge[current][neighbor][0]['weight'])
            if G.node[neighbor]['max_weight'] < m:
                # We can reach the neighbor node from the current node
                # through a path with a wider bottleneck that the one
                # we (perhaps) already knew.
                G.node[neighbor]['max_weight'] = m
                G.node[neighbor]['predecessor'] = current

                # Update the location of the bottleneck.
                if weight < m:
                    G.node[neighbor]['bottleneck'] \
                        = G.node[current]['bottleneck']
                else:
                    G.node[neighbor]['bottleneck'] = (current, neighbor)


def compute_global_max_weight_path(G: networkx.MultiDiGraph,
                                   start: int = META_REACTANT,
                                   stop: int = META_PRODUCT) \
        -> List[Tuple[int, int]]:
    """Compute global maximum weight path.

    """
    if start == stop:
        return []

    compute_max_weight(G, start)

    bottleneck = G.node[stop]['bottleneck']

    return compute_global_max_weight_path(G, start, bottleneck[0]) \
        + [(bottleneck[0], bottleneck[1])] \
        + compute_global_max_weight_path(G, stop, bottleneck[1])


def max_weight_path(transition_matrix: scipy.sparse.coo_matrix,
                    stationary_vector: np.array,
                    reactants: np.array, products: np.array) -> np.array:
    """Compute maximum weight path.

    This program computes global maximum weight paths using flux-space
    graphs with flux-based edge weights (see [1]).

    Parameters
    ----------
    transition_matrix : scipy.sparse.coo_matrix
        Transition matrix for the system.
    stationary_vector : np.array
        Vector of stationary flux weights.
    reactants : np.array
        Array of indices corresponding to the reactant milestones.
    products : np.array
        Array of indices corresponding to the products milestones.

    Returns
    -------
    p : np.array
        Vector containing the sequence of steps through the maximum
        weight path. Entries corresponding to milestones involved in
        the path contain the step number at which they occurr. All
        other entries contain NaN.

    Notes
    -----
    There is a modification to the algorithm in [1]: we extend the
    graph by adding two extra nodes. One node that acts as a source
    and another node that acts as a sink. The source is connected to
    the reactant milestones through edges weighted according to the
    fluxes of the reactants. The product milestones are connected to
    the sink via edges with zero weight.

    .. [1] Viswanath, S., Kreuzer, S. M., Cardenas, A. E., & Elber, R. (2013). Analyzing milestoning networks for molecular kinetics: definitions, algorithms, and examples. The Journal of Chemical Physics, 139(17), 174105. doi:10.1063/1.4827495

    """
    G = make_connectivity_graph(transition_matrix, stationary_vector,
                                reactants, products)

    logging.debug('Computing global maximum weight path...')
    sequence = compute_global_max_weight_path(G)
    logging.debug('Done.')

    p = np.empty(transition_matrix.shape[0])
    p.fill(np.nan)

    for i, nodes in enumerate(sequence[1:]):
        idx = nodes[0]-1
        p[idx] = i

    return p
