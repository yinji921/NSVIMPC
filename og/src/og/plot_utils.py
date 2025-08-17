import math

import ipdb
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from og.none import get_or
from og.optim.nnls import nnls


def padded_minmax(arr, pad_frac: float = 0.02, min_width: float = None):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    arr_min, arr_max = arr.min(), arr.max()
    ptp = arr_max - arr_min
    pad = ptp * pad_frac
    ymin, ymax = arr_min - pad, arr_max + pad

    if min_width is not None:
        if ymax - ymin < min_width:
            mid = (ymax + ymin) / 2
            ymin, ymax = mid - min_width / 2, mid + min_width / 2

    return ymin, ymax


def _move_min_distance(targets: ArrayLike, min_distance: float) -> np.ndarray:
    """Move the targets such that they are close to their original positions, but keep
    min_distance apart.

    https://math.stackexchange.com/a/3705240/36678
    """
    # sort targets
    idx = np.argsort(targets)
    targets = np.sort(targets)

    n = len(targets)
    x0_min = targets[0] - n * min_distance
    A = np.tril(np.ones([n, n]))
    b = targets - (x0_min + np.arange(n) * min_distance)

    # import scipy.optimize
    # out, _ = scipy.optimize.nnls(A, b)

    out = nnls(A, b)

    sol = np.cumsum(out) + x0_min + np.arange(n) * min_distance

    # reorder
    idx2 = np.argsort(idx)
    return sol[idx2]


def line_labels(
    ax: plt.Axes | None = None,
    min_label_distance: float | str = "auto",
    alpha: float = 1.0,
    label_pos: str = "right",
    label_eps: float = 1e-3,
    **text_kwargs,
):
    if ax is None:
        ax = plt.gca()

    logy = ax.get_yscale() == "log"

    if min_label_distance == "auto":
        # Make sure that the distance is alpha * fontsize. This needs to be translated
        # into axes units.
        fig_height_inches = plt.gcf().get_size_inches()[1]
        ax_pos = ax.get_position()
        ax_height = ax_pos.y1 - ax_pos.y0
        ax_height_inches = ax_height * fig_height_inches
        ylim = ax.get_ylim()
        if logy:
            ax_height_ylim = math.log10(ylim[1]) - math.log10(ylim[0])
        else:
            ax_height_ylim = ylim[1] - ylim[0]
        # 1 pt = 1/72 in
        fontsize = plt.rcParams["font.size"]
        assert fontsize is not None
        min_label_distance_inches = fontsize / 72 * alpha
        min_label_distance = min_label_distance_inches / ax_height_inches * ax_height_ylim

    # find all Line2D objects with a valid label and valid data
    lines = [
        child
        for child in ax.get_children()
        # https://stackoverflow.com/q/64358117/353337
        if (isinstance(child, plt.Line2D) and child.get_label()[0] != "_" and not np.all(np.isnan(child.get_ydata())))
    ]

    if len(lines) == 0:
        return

    if label_pos == "right":
        ypos_idx = -1
    elif label_pos == "left":
        ypos_idx = 0
    else:
        raise ValueError("label_pos must be 'left' or 'right'")

    # Add "legend" entries.
    # Get last non-nan y-value.
    targets = []
    for line in lines:
        ydata = np.array(line.get_ydata())
        targets.append(ydata[~np.isnan(ydata)][ypos_idx])

    if logy:
        targets = [math.log10(t) for t in targets]

    # Sometimes, the max value if beyond ymax. It'd be cool if in this case we could put
    # the label above the graph (instead of the to the right), but for now let's just
    # cap the target y.
    ymax = ax.get_ylim()[1]
    targets = [min(target, ymax) for target in targets]

    targets = _move_min_distance(targets, min_label_distance)
    if logy:
        targets = [10**t for t in targets]

    labels = [line.get_label() for line in lines]
    colors = [line.get_color() for line in lines]

    # Leave the labels some space to breathe. If they are too close to the
    # lines, they can get visually merged.
    # <https://twitter.com/EdwardTufte/status/1416035189843714050>
    # Don't forget to transform to axis coordinates first. This makes sure the
    # https://stackoverflow.com/a/40475221/353337
    axis_to_data = ax.transAxes + ax.transData.inverted()
    if label_pos == "right":
        xpos = axis_to_data.transform([1.00 + label_eps, 1.0])[0]
        text_kwargs["ha"] = "left"
    elif label_pos == "left":
        xpos = axis_to_data.transform([-0.00 - label_eps, 1.0])[0]
        text_kwargs["ha"] = "right"
    else:
        raise ValueError("label_pos must be 'left' or 'right'")

    for label, ypos, color in zip(labels, targets, colors):
        ax.text(xpos, ypos, label, verticalalignment="center", color=color, **text_kwargs)


def axvline_labeled(
    ax: plt.Axes, xpos: float, text: str, xytext, axvline_kwargs: dict = None, text_kwargs: dict = None
):
    axvline_kwargs = get_or(axvline_kwargs, {})
    text_kwargs = get_or(text_kwargs, {})

    # trans = ax.get_xaxis_transform()

    line = ax.axvline(xpos, **axvline_kwargs)
    # text = ax.text(xpos, 1.0, text, transform=trans, **text_kwargs)
    text = ax.annotate(
        text,
        (xpos, 1.0),
        xytext=xytext,
        xycoords=("data", "axes fraction"),
        textcoords="offset points",
        **text_kwargs,
    )
    return line, text
