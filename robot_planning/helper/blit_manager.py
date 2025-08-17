import ipdb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colorbar import Colorbar


class BlitManager:
    def __init__(self, canvas: FigureCanvasAgg, animated_artists=(), animated_cbars=()):
        self.canvas = canvas
        self._bg = None
        self._artists: list[plt.Artist] = []
        self._cbars: list[Colorbar] = []

        for a in animated_artists:
            self.add_artist(a)

        for cbar in animated_cbars:
            self.add_cbar(cbar)

        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art: plt.Artist):
        try:
            if art.figure != self.canvas.figure:
                raise RuntimeError
        except AttributeError:
            import traceback
            traceback.print_exc()  # 👈 prints full error message
            ipdb.set_trace()
        art.set_animated(True)
        self._artists.append(art)

    def add_cbar(self, cbar: Colorbar):
        if cbar.ax.figure != self.canvas.figure:
            raise RuntimeError

        for artist in cbar.ax.get_children():
            artist.set_animated(True)
        self._cbars.append(cbar)

    def _draw_animated(self, mappables=None):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

        if mappables is None:
            return

        assert len(mappables) == len(self._cbars)
        for mappable, cbar in zip(mappables, self._cbars):
            cbar.update_normal(mappable)
            for a in cbar.ax.get_children():
                fig.draw_artist(a)
            fig.draw_artist(cbar.solids)

    def update(self, mappables=()):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated(mappables)
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()
