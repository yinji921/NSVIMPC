import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path


def plot_x_bounds(ax: plt.Axes, bounds: tuple[float | None, float | None], obs_style: dict):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xlen, ylen = xmax - xmin, ymax - ymin
    lb, ub = bounds

    # Lower bounds.
    if lb is not None:
        rect = plt.Rectangle((xmin, ymin), lb - xmin, ylen, **obs_style)
        ax.add_patch(rect)

    # Upper bounds.
    if ub is not None:
        rect = plt.Rectangle((ub, ymin), xmax - ub, ylen, **obs_style)
        ax.add_patch(rect)


def plot_y_bounds(ax: plt.Axes, bounds: tuple[float, float], obs_style: dict):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xlen, ylen = xmax - xmin, ymax - ymin
    lb, ub = bounds

    # Lower bounds.
    rect = plt.Rectangle((xmin, ymin), xlen, lb - ymin, **obs_style)
    ax.add_patch(rect)
    # Upper bounds.
    rect = plt.Rectangle((xmin, ub), xlen, ymax - ub, **obs_style)
    ax.add_patch(rect)


def plot_x_goal(ax: plt.Axes, bounds: tuple[float, float], goal_style: dict):
    ymin, ymax = ax.get_ylim()
    ylen = ymax - ymin
    lb, ub = bounds

    rect = plt.Rectangle((lb, ymin), ub - lb, ylen, **goal_style)
    ax.add_patch(rect)


def poly_to_patch(poly, **kwargs) -> PathPatch:
    import shapely

    assert isinstance(poly, shapely.Polygon)
    ext_path = np.asarray(poly.exterior.coords)[:, :2]
    path = Path.make_compound_path(Path(ext_path), *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    return patch
