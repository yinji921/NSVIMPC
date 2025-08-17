import matplotlib
import numpy as np

from og.none import get_or


def blend_color_with_white(color, alpha: float):
    """Get the resulting color if `color` has transparency `alpha` and is on a white background."""
    # 1: Convert color to RGB. Values between 0 and 1.
    color = np.array(matplotlib.colors.to_rgb(color))

    # 2: Blend with white.
    white = np.array([1.0, 1.0, 1.0])
    blended = (1 - alpha) * white + alpha * color
    return blended


def modify_hsv(color, h: float = None, s: float = None, v: float = None):
    # 1: Convert color to hsv.
    color = np.array(matplotlib.colors.to_rgb(color))
    hsv = matplotlib.colors.rgb_to_hsv(color)
    # 2: Modify hsv.
    hsv[0] += get_or(h, 0.0)
    hsv[1] += get_or(s, 0.0)
    hsv[2] += get_or(v, 0.0)
    # 3: Clip color.
    hsv = np.clip(hsv, 0.0, 1.0)
    # 4: Convert back to rgb.
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return rgb
