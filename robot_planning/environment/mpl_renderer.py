import ast
import pdb
import time
from timeit import default_timer
from matplotlib.lines import Line2D
import einops as ei
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import ColormapRegistry
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap

from robot_planning.helper.blit_manager import BlitManager
from robot_planning.environment.dynamics.autorally_dynamics.map_coords import MapCA
from robot_planning.environment.renderers import Renderer
from robot_planning.helper.utils import AUTORALLY_DYNAMICS_DIR
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import dynamics_factory_base


def register_cmaps():
    sns_cmaps = ["rocket", "mako", "flare", "crest", "vlag", "icefire"]

    colormaps: ColormapRegistry = matplotlib.colormaps

    for cmap_name in sns_cmaps:
        if cmap_name in plt.colormaps():
            continue

        cmap = sns.color_palette(cmap_name, as_cmap=True)
        colormaps.register(cmap, name=cmap_name)

    # Construct cubic versions of all the colormaps.
    for cmap_name in sns_cmaps:
        for suffix in ["", "_r"]:
            name = cmap_name + suffix + "_quadratic"
            if name in plt.colormaps():
                continue

            cmap = plt.get_cmap(cmap_name + suffix)
            idxs = np.linspace(0, 1, 2 * cmap.N)
            colors = cmap(idxs**2)
            cmap_new = ListedColormap(colors, name=name)
            colormaps.register(cmap_new)

        for suffix in ["", "_r"]:
            name = cmap_name + suffix + "_cubic"
            if name in plt.colormaps():
                continue

            cmap = plt.get_cmap(cmap_name + suffix)
            idxs = np.linspace(0, 1, 2 * cmap.N)
            colors = cmap(idxs**3)
            cmap_new = ListedColormap(colors, name=name)
            colormaps.register(cmap_new)


class MutablePatchCollection(PatchCollection):
    def __init__(self, patches, *args, **kwargs):
        self.patches = patches
        self._paths = None
        PatchCollection.__init__(self, patches, *args, **kwargs)

    def set_patches(self, patches):
        self.patches = patches
        self.stale = True

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths


class AutorallyMatplotlibRenderer(Renderer):
    """Use blitting for faster rendering."""

    def __init__(self, active: bool = True):
        super().__init__()
        register_cmaps()
        self.xaxis_range: float | None = None
        self.yaxis_range: float | None = None
        self.figure_size: float | None = None
        self.figure_dpi: float | None = None

        self.trajectories_rendering: bool | None = None
        self.map = None
        self.track: MapCA | None = None
        self.track_widths: list[float] | None = None
        self.path_rendering = None

        self._figure: plt.Figure | None = None
        self._ax: plt.Axes | None = None
        self._bm: BlitManager | None = None

        self.path = []
        self.active = active

        self.frame = 0

        ##################################################
        # self.agent_circle
        self.goal = None
        self.track_lines: list[plt.Line2D] | None = None

        self.robot_circ: plt.Circle | None = None
        self.pcm: plt.PathCollection | None = None
        self.cbar: plt.Colorbar | None = None

        self.traj_col: LineCollection | None = None
        self.opt_traj: plt.Line2D | None = None
        self.obs_col: MutablePatchCollection | None = None
        ##################################################
        # self.sleep_time = 0.02
        self.sleep_time = 0.0

    def initialize_from_config(self, config_data, section_name):
        self.xaxis_range = np.asarray(
            ast.literal_eval(config_data.get(section_name, "xaxis_range")),
            dtype=np.float64,
        )
        self.yaxis_range = np.asarray(
            ast.literal_eval(config_data.get(section_name, "yaxis_range")),
            dtype=np.float64,
        )
        self.figure_size = np.asarray(ast.literal_eval(config_data.get(section_name, "figure_size")), dtype=int)
        self.figure_dpi = config_data.getint(section_name, "figure_dpi")

        if config_data.has_option(section_name, "trajectories_rendering"):
            self.trajectories_rendering = config_data.getboolean(section_name, "trajectories_rendering")

        if config_data.has_option(section_name, "map_file"):
            map_file = config_data.get(section_name, "map_file")
            self.map = np.load(AUTORALLY_DYNAMICS_DIR + "/" + map_file)
        if config_data.has_option(section_name, "path_rendering"):
            self.path_rendering = config_data.get(section_name, "path_rendering")

        if config_data.has_option(section_name, "track_file_name"):
            track_file_name = config_data.get(section_name, "track_file_name")
            track_path = AUTORALLY_DYNAMICS_DIR + "/" + track_file_name
            self.track = MapCA(track_path)
            track_widths = config_data.get(section_name, "track_widths")
            self.track_widths = np.asarray(ast.literal_eval(track_widths), dtype=np.float64)


    def _create_figure(self) -> tuple[plt.Figure, plt.Axes]:
        fig = plt.figure(figsize=(self.figure_size[0], self.figure_size[1]), dpi=self.figure_dpi)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal")
        return fig, ax

    def _get_figax(self):
        if self._figure is None:
            self._figure, self._ax = self._create_figure()

        return self._figure, self._ax

    def close(self):
        plt.close("all")

    def render_states(self, state_list=None, kinematics_list=None, **kwargs):
        if not self.active:
            return

        fig, ax = self._get_figax()

        assert len(state_list) == len(kinematics_list) == 1
        state, kinematics = state_list[0], kinematics_list[0]

        if self.robot_circ is None:
            self.robot_circ = plt.Circle((state[-2], state[-1]), kinematics.radius, **kwargs)
            ax.add_patch(self.robot_circ)

        self.robot_circ.center = (state[-2], state[-1])
        self.robot_circ.radius = kinematics.radius

        # kinematics = kinematics_list[ii]
        # circle = plt.Circle((state[-2], state[-1]), kinematics.radius, **kwargs)
        # ax.add_patch(circle)

        if not self.path_rendering:
            return

        pos2d = np.stack([state[0], state[6], state[7]])
        self.path.append(pos2d)
        path = np.stack(self.path, axis=1)

        if self.pcm is None:
            cmap1 = plt.get_cmap("flare_r_quadratic")(np.linspace(0.0, 1.0, 256 // 2))
            cmap2 = plt.get_cmap("turbo")(np.linspace(0.1, 0.4, 8 // 2))
            colors = np.concatenate([cmap1, cmap2], axis=0)
            cmap = LinearSegmentedColormap.from_list('my_colormap', colors)
            self.pcm = ax.scatter(path[1, :], path[2, :], c=path[0, :], cmap=cmap, marker=".")
            self.cbar = fig.colorbar(self.pcm)
            self.cbar.set_label("speed (m/s)")

        self.pcm.set_offsets(path[1:3, :].T)
        self.pcm.set_array(path[0, :])
        vmin, vmax = path[0, :].min(), path[0, :].max()
        norm = Normalize(vmin, vmax)
        self.pcm.set_norm(norm)

    def render_obstacles(self, obstacle_list=None, **kwargs):
        # assert len(obstacle_list) == 0
        fig, ax = self._get_figax()
        map_style = dict(color="0.6", alpha=0.3)
        center_style = dict(color="0.5", lw=0.8, alpha=0.3)
        track_style = dict(color="#E24A33", lw=0.8, alpha=0.7)

        # # TODO Eric
        # map_style = dict(color="0.6", alpha=0)
        # center_style = dict(color="0.5", lw=1.5, alpha=0.5)
        # track_style = dict(color="#E24A33", lw=2, alpha=1.0)

        if self.track_lines is None:
            (line_in,) = ax.plot(self.map["X_in"], self.map["Y_in"], **map_style)
            (line_out,) = ax.plot(self.map["X_out"], self.map["Y_out"], **map_style)

            track_lines = []
            # Plot centerline.
            (line,) = ax.plot(self.track.midpoints[0], self.track.midpoints[1], **center_style)
            track_lines.append(line)
            # # TODO Eric
            # for width in [self.track_widths[0]]:
            for width in self.track_widths:
                for w_ in [width, -width]:
                    xs = self.track.midpoints[0] + np.cos(self.track.heading + np.pi / 2) * w_
                    ys = self.track.midpoints[1] + np.sin(self.track.heading + np.pi / 2) * w_
                    # Make it a closed line by adding the first point to the end.
                    xs = np.append(xs, xs[0])
                    ys = np.append(ys, ys[0])
                    (line,) = ax.plot(xs, ys, **track_style)
                    track_lines.append(line)

            self.track_lines = [line_in, line_out, *track_lines]

        if self.obs_col is None:
            circs = []
            for x, y, size in obstacle_list:
                circle = plt.Circle((x, y), size, **kwargs)
                circs.append(circle)
            self.obs_col = MutablePatchCollection(circs, match_original=True)
            ax.add_collection(self.obs_col)
        else:
            circs = []
            for x, y, size in obstacle_list:
                circle = plt.Circle((x, y), size, **kwargs)
                circs.append(circle)
            self.obs_col.set_patches(circs)

    def render_goal(self, goal=None, **kwargs):
        if self.goal is not None:
            return

        fig, ax = self._get_figax()
        x = goal[0]
        y = goal[1]
        radius = goal[2]
        self.goal = plt.Circle((x, y), radius, **kwargs, zorder=0)
        ax.add_patch(self.goal)

    def render_trajectories(self, trajectory_list=None, **kwargs):
        if not self.trajectories_rendering:
            return

        fig, ax = self._get_figax()
        is_opt_traj = len(trajectory_list) == 1
        if is_opt_traj:
            xT_state = np.array(trajectory_list[0])
            T_pos = xT_state[-2:, :].T

            if self.opt_traj is None:
                (self.opt_traj,) = ax.plot(T_pos[:, 0], T_pos[:, 1], color="C1", lw=1.2, zorder=6)
                # # TODO Eric
                # (self.opt_traj,) = ax.plot(T_pos[:, 0], T_pos[:, 1], color="C1", lw=1.5, zorder=6)

            self.opt_traj.set_data(T_pos[:, 0], T_pos[:, 1])
        else:
            bxT_state = np.array(trajectory_list)
            b2T_pos = bxT_state[:, -2:, :]
            bT2_pos = ei.rearrange(b2T_pos, "b nx T -> b T nx")

            if self.traj_col is None:
                self.traj_col = LineCollection(bT2_pos, colors="C2", lw=0.2, alpha=0.4, zorder=5)
                # # TODO Eric
                # self.traj_col = LineCollection(bT2_pos, colors="C2", lw=0.5, alpha=0.05, zorder=5)
                ax.add_collection(self.traj_col)

            self.traj_col.set_segments(bT2_pos)

    @property
    def animated_artists(self):
        artists = [self.robot_circ, self.pcm, self.obs_col]
        if self.traj_col is not None:
            artists.append(self.traj_col)
        if self.opt_traj is not None:
            artists.append(self.opt_traj)
        return artists

    def show(self):
        assert self.active

        if self._bm is None:
            # Initialize the blit manager.
            self._bm = BlitManager(self._figure.canvas, self.animated_artists, [self.cbar])
            # make sure our window is on the screen and drawn
            plt.show(block=False)
            plt.pause(0.01)

        self._bm.update([self.pcm])
        self.frame += 1

        if self.sleep_time > 0:
            start = default_timer()
            end = start + self.sleep_time
            while default_timer() < end:
                self._bm.canvas.flush_events()
                time.sleep(0.005)

    def clear(self):
        pass


