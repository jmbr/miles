"""Module for plotting routines.

"""


__all__ = ['plot']

import os
import sys
from typing import Optional, Tuple

import scipy.ndimage
import scipy.linalg
import scipy.spatial

matplotlib = None               # type: Optional[module]
plt = None                      # type: Optional[module]
sns = None                      # type: Optional[module]

import numpy as np

from miles import Milestones, Simulation, load_distributions  # noqa: E501


EPSILON = sys.float_info.epsilon

TICKS = [-180, -90, 0, 90, 180]


def latex_preamble() -> None:
    """LaTeX preamble for publication-quality figures.

    """
    fig_width_pt = 397.48499
    inches_per_pt = 1.0/72.27
    golden_mean = (np.sqrt(5)-1.0) / 2.0
    fig_width = 1.4 * fig_width_pt * inches_per_pt
    fig_height = fig_width * 1.25 * golden_mean
    fig_size = (fig_width, fig_height)

    params = {
        'backend': 'pdf',
        'pgf.texsystem': 'pdflatex',
        'font.family': 'serif',
        'font.serif': 'cm',
        'font.size': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'text.usetex': True,
        'text.latex.unicode': True,
        'figure.figsize': fig_size
    }
    matplotlib.rcParams.update(params)

    plt.axes([0.2, 0.2, 1.0 - 0.2, 1.0 - 0.2])

    plt.figure(figsize=fig_size)


def set_xticks() -> None:
    plt.axes().set_xticks(TICKS)
    plt.axes().set_xlim([TICKS[0], TICKS[-1]])


def set_yticks() -> None:
    plt.axes().set_yticks(TICKS)
    plt.axes().set_ylim([TICKS[0], TICKS[-1]])


def pbc(points: np.array) -> np.array:
    """Apply periodic boundary conditions to a set of points.

    """
    return np.concatenate((points,
                           points + np.array([0, -360]),
                           points + np.array([0, 360]),
                           points + np.array([-360, -360]),
                           points + np.array([-360, 0]),
                           points + np.array([-360, +360]),
                           points + np.array([360, -360]),
                           points + np.array([360, 0]),
                           points + np.array([360, 360])))


def plot_voronoi(vor: scipy.spatial.Voronoi, milestones: Milestones,
                 data: np.array, ax: 'matplotlib.axes._axes.Axes', **kwargs):
    """Plot 2D Voronoi diagram.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        Voronoi tessellation to plot.
    milestones : Milestones
        Milestone factory.
    data : np.array
        Dataset to plot onto the milestones.
    ax : matplotlib.axes._axes.Axes
        Axes where the figure will be plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure.

    """
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    if data is None:
        data = np.ones(milestones.max_milestones)
        plot_colorbar = False
    else:
        plot_colorbar = True

    import matplotlib.cm as cm

    cmap = cm.get_cmap(kwargs['colors'])

    valid_elements = np.logical_not(np.isnan(data))
    minimum = np.min(data[valid_elements])
    maximum = np.max(data[valid_elements])
    norm = cm.colors.Normalize(vmin=minimum, vmax=maximum)

    alpha = 0.9
    sz = kwargs['marker_size']
    labels = kwargs['labels']

    num_anchors = len(milestones.anchors)

    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            try:
                m = milestones.make_from_indices(pointidx[0] % num_anchors,
                                                 pointidx[1] % num_anchors)
            except IndexError:
                continue

            if labels:
                point = vor.vertices[simplex, :].mean(axis=0)
                ax.text(point[0], point[1], [a.index for a in m.anchors],
                        clip_on=True, size=sz, zorder=10)

            datum = data[m.index]
            if not np.isnan(datum):
                c = cmap(norm(datum))
                ax.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1],
                        color=c, alpha=alpha, linewidth=sz, zorder=-10+datum)

    length_far_point = 1e6      # XXX hard-coded

    center = vor.points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + length_far_point * direction

            try:
                m = milestones.make_from_indices(pointidx[0] % num_anchors,
                                                 pointidx[1] % num_anchors)
            except IndexError:
                continue

            if labels:
                point = vor.vertices[i] + direction * 0.5  # XXX hardcoded
                ax.text(point[0], point[1], [a.index for a in m.anchors],
                        clip_on=True, size=sz, zorder=10)

            datum = data[m.index]
            if not np.isnan(datum):
                c = cmap(norm(datum))
                ax.plot([vor.vertices[i, 0], far_point[0]],
                        [vor.vertices[i, 1], far_point[1]], color=c,
                        alpha=alpha, linewidth=sz, zorder=-10+datum)

    if plot_colorbar:
        ax1, _ = matplotlib.colorbar.make_axes(ax)
        cb = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm)
        if kwargs['colorbar_title']:
            cb.set_label(kwargs['colorbar_title'])
        text = cb.ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(style='normal')
        text.set_font_properties(font)

    return ax.figure


def plot_free_energy():
    # from matplotlib.colors import LogNorm
    ala3 = np.load('ala3.npy')
    d = ala3[:, [1, -1]]
    num_bins = 50
    zz, xx, yy = scipy.histogram2d(d[:, 0], d[:, 1], bins=num_bins,
                                   normed=True)
    x, y = np.meshgrid(xx[:-1], yy[:-1])
    z = scipy.ndimage.gaussian_filter(zz, sigma=1.0, order=0)
    plt.contour(y, x, z, colors='k', zorder=-900)
    plt.imshow(z.T, extent=(-180, 180, -180, 180), origin='lower',
               cmap='Greys', zorder=-1000)
    # cmap='YlGnBu', zorder=-1000)
    # plt.colorbar(label='Density')


def plot_milestones(simulation: Simulation, **kwargs) -> None:
    """Plot a dataset on the faces of a Voronoi tessellation.

    """
    milestones = simulation.milestones
    anchors = milestones.anchors
    assert len(anchors) >= 2

    if kwargs['input']:
        input_files = kwargs['input']
        input_file_name = input_files[0]
        try:
            data = np.load(input_file_name)
        except OSError:
            data = np.loadtxt(input_file_name)
    else:
        data = None

    sns.set(style='ticks')

    if kwargs['output']:
        latex_preamble()

    plot_free_energy()

    coordinates = np.array([a.coordinates for a in anchors])
    voronoi = scipy.spatial.Voronoi(pbc(coordinates))  # XXX PBC hardcoded
    # voronoi = scipy.spatial.Voronoi(coordinates, qhull_options='Qp Pp')

    plot_voronoi(voronoi, milestones, data, ax=plt.axes(), **kwargs)

    set_xticks()
    set_yticks()

    plt.axes().set_aspect('equal')

    if kwargs['title']:
        plt.title(kwargs['title'])
    if kwargs['xlabel']:
        plt.xlabel(kwargs['xlabel'])
    if kwargs['ylabel']:
        plt.ylabel(kwargs['ylabel'])

    if kwargs['output']:
        plt.savefig(kwargs['output'])
    else:
        plt.show()


def orthogonal_distance_regression(data: np.array) \
        -> Tuple[np.array, np.array]:
    """Fit data to best hyperplane (best in the least-squares sense).

    Parameters
    ----------
    data : np.array
        Points to fit.

    Returns
    -------
    projected_points : np.array
        The points projected onto the best hyperplane.
    coordinates : np.array
        The (n-1) coordinates of the points on the best hyperplane.

    """
    d = np.array(data, dtype=data.dtype)
    center_of_mass = np.mean(d, axis=0)
    d -= center_of_mass

    U, D, V = scipy.linalg.svd(d.T, full_matrices=False)

    # Subspace spanned by the best hyperplane.
    span = U[:, 0:-1]

    # Projection of the data points onto the best hyperplane.
    projection = np.matrix(center_of_mass) + d @ span @ span.T

    # Coordinates of the data points in the hyperplane. Note that
    # since U is an orthogonal matrix, its inverse coincides with its
    # transpose.
    d += center_of_mass
    coordinates = -U.T @ d.T

    return np.array(projection), coordinates[0:-1, :].T


def sample_data(distribution, key, num_samples=1000):  # XXX Hardcoded.
    """Draw samples from a given distribution.

    """
    data = []

    for _ in range(num_samples):
        transition = distribution.sample()
        data.append(key(transition))

    return np.array(data)


def get_rows_and_columns(num_milestones: int) -> Tuple[int, int]:
    """Get optimal number of rows and columns to display histograms.

    Parameters
    ----------
    num_milestones : int
        Number of milestones

    Returns
    -------
    rows : int
        Optimal number of rows.
    cols : int
        Optimal number of columns.

    """
    if num_milestones <= 10:
        layouts = {
            1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3),
            6: (2, 3), 7: (2, 4), 8: (2, 4), 9: (3, 9), 10: (2, 5)
        }
        rows, cols = layouts[num_milestones]
    else:
        rows = int(np.ceil(np.sqrt(num_milestones)))
        cols = rows

    return rows, cols


def plot_histograms(simulation: Simulation, **kwargs) -> None:
    """Plot histograms on milestones.

    """
    sns.set(style='ticks')

    if kwargs['output'] is not None:
        latex_preamble()

    all_distributions = []
    known_milestones = set()

    file_names = kwargs['input']
    num_bins = kwargs['num_bins']
    min_value, max_value = kwargs['min_value'], kwargs['max_value']

    for file_name in file_names:
        dists = load_distributions(file_name)
        all_distributions.append(dists)
        known_milestones = known_milestones.union({m for m in dists.keys()})

    plot_kde = len(file_names) > 1
    if num_bins:
        pts = np.linspace(-180, 180, num_bins)  # XXX hardcoded.
        bins = pts[:-1]
        plot_hist = True
    else:
        bins = None
        plot_hist = not plot_kde

    rows, cols = get_rows_and_columns(len(known_milestones))

    for idx, milestone in enumerate(sorted(known_milestones)):
        for file_name, dists in zip(file_names, all_distributions):
            try:
                distribution = dists[milestone]
            except KeyError:
                continue

            data = sample_data(distribution,
                               key=lambda x: x.colvars)
            _, x = orthogonal_distance_regression(data)
            # x = data[:, 1]      # XXX BUG

            plt.subplot(rows, cols, idx + 1)
            _, name = os.path.split(file_name)
            sns.distplot(x, hist=plot_hist, norm_hist=True,
                         kde=plot_kde, bins=bins, label=name)
            sns.despine()

            plt.xlabel('Position at milestone')
            plt.xlim([TICKS[0], TICKS[-1]])
            plt.xticks(TICKS)

            plt.ylabel('Density')
            if min_value is not None and max_value is not None:
                plt.ylim([min_value, max_value])
            plt.yticks([])

            plt.legend()
            plt.title(milestone)

    plt.tight_layout()

    if kwargs['output']:
        plt.savefig(kwargs['output'])
    else:
        plt.show()


def import_modules() -> None:
    """Import slow-loading modules."""
    global matplotlib
    import matplotlib
    # matplotlib.use('pgf')

    global plt
    import matplotlib.pyplot as plt

    global sns
    import seaborn as sns


def plot(simulation: Simulation, **kwargs) -> None:
    """Plot simulation results.

    Parameters
    ----------
    simulation : Simulation
        Object containing all the relevant info about the simulation.

    """
    import_modules()

    # cv = kwargs['colvars']
    # colvars_spec = [int(c) for c in cv.strip().split(',')]
    #
    # print(colvars_spec)
    #
    # collective_variables = simulation.collective_variables
    #
    # cvs = []
    # for i, cv in enumerate(collective_variables.collective_variables):
    #     cvs.append(cv)

    if kwargs['histograms']:
        plot_histograms(simulation, **kwargs)
    else:
        plot_milestones(simulation, **kwargs)
