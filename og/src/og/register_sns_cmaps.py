import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger as log
from matplotlib.cm import ColormapRegistry


def register_cmaps():
    sns_cmaps = ["rocket", "mako", "flare", "crest", "vlag", "icefire"]

    for cmap_name in sns_cmaps:
        if cmap_name in plt.colormaps():
            continue

        cmap = sns.color_palette(cmap_name, as_cmap=True)
        colormaps: ColormapRegistry = matplotlib.colormaps
        colormaps.register(cmap, name=cmap_name)
        log.info("Registered colormap {}".format(cmap_name))
