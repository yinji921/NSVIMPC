import ast
import loguru
import time
from timeit import default_timer

import einops as ei
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text

from robot_planning.helper.blit_manager import BlitManager
from robot_planning.helper.convenience import get_dubins_kinematics
from robot_planning.environment.mpl_renderer import MutablePatchCollection, register_cmaps
from robot_planning.environment.renderers import Renderer
from robot_planning.factory.factories import dynamics_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.helper.utils import AUTORALLY_DYNAMICS_DIR

class DubinsMatplotlibRendererBlit(Renderer):
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

        # Assuming the Dubins car is a cirle.
        self.dubins_com: plt.Circle | None = None

        self.pcm: plt.PathCollection | None = None
        self.cbar: plt.Colorbar | None = None

        self.traj_col: LineCollection | None = None
        self.opt_traj: plt.Line2D | None = None
        self.safe_traj: plt.Line2D | None = None
        self.obs_col: MutablePatchCollection | None = None

        self.title_text: Text | None = None
        ##################################################
        # self.sleep_time = 0.02
        self.sleep_time = 0.0

        self.kin = get_dubins_kinematics()

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

        if config_data.has_option(section_name, "path_rendering"):
            self.path_rendering = config_data.get(section_name, "path_rendering")

    def _create_figure(self) -> tuple[plt.Figure, plt.Axes]:
        fig = plt.figure(figsize=(self.figure_size[0], self.figure_size[1]), dpi=self.figure_dpi)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal")
        return fig, ax

    def _get_figax(self):
        if self._figure is None:
            self._figure, self._ax = self._create_figure()

        self._ax.set_xlim(self.xaxis_range)
        self._ax.set_ylim(self.yaxis_range)

        return self._figure, self._ax

    def close(self):
        plt.close("all")

    def set_dubins_artists(self, state: np.ndarray):
        kinematics = self.kin

        x, y, theta = state[:3]
        com_pos = np.array([x, y])
        self.dubins_com.set_center(com_pos)
        self.dubins_com.set_radius(kinematics.get_radius())

    def create_dubins_artists(self):
        assert self.dubins_com is None
        kinematics = self.kin

        half_body_length_to_rotor_size_ratio = 2
        radius = kinematics.get_radius()

        color_com = "black"

        self.dubins_com = plt.Circle((0, 0), radius, facecolor=color_com, edgecolor="none", zorder=5.5)

        self.title_text = self._ax.set_title("")

        self._ax.add_artist(self.dubins_com)

    def render_states(self, state_list=None, kinematics_list=None, **kwargs):
        if not self.active:
            return

        fig, ax = self._get_figax()

        assert len(state_list) == len(kinematics_list) == 1
        state, kinematics = state_list[0], kinematics_list[0]

        if self.dubins_com is None:
            self.create_dubins_artists()

        self.set_dubins_artists(state)

        ## Update the current title with the velocity info.
        # vel = np.linalg.norm(state[3:5])
        # text = r"vel: [ ${:+.1f}$, ${:+.1f}$ ]   ( ${:+.1f}$".format(state[3], state[4], vel)
        text = "Constant Speed"
        self.title_text.set_text(text)

        if not self.path_rendering:
            return

        # [px, py]
        pos2d = np.stack([state[3], state[0], state[1]])
        self.path.append(pos2d)
        path = np.stack(self.path, axis=1)

        if self.pcm is None:
            cmap1 = plt.get_cmap("flare_r_quadratic")(np.linspace(0.0, 1.0, 256 // 2))
            cmap2 = plt.get_cmap("turbo")(np.linspace(0.1, 0.4, 8 // 2))
            colors = np.concatenate([cmap1, cmap2], axis=0)
            cmap = LinearSegmentedColormap.from_list("my_colormap", colors)
            self.pcm = ax.scatter(path[1, :], path[2, :], c=path[0, :], cmap=cmap, marker=".")
            self.cbar = fig.colorbar(self.pcm)
            self.cbar.set_label("steering angle (rad)")

        self.pcm.set_offsets(path[1:3, :].T)
        self.pcm.set_array(path[0, :])
        vmin, vmax = path[0, :].min(), path[0, :].max()
        norm = Normalize(vmin, vmax)
        self.pcm.set_norm(norm)

    def render_obstacles(self, obstacle_list=None, **kwargs):
        # assert len(obstacle_list) == 0
        fig, ax = self._get_figax()

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

        kwargs = dict(kwargs)

        fig, ax = self._get_figax()
        is_opt_traj = len(trajectory_list) == 1
        if is_opt_traj:
            xT_state = np.array(trajectory_list[0])
            T_pos = xT_state[:2, :].T

            if "trajtype" in kwargs and kwargs["trajtype"] == "vsafe":
                if self.safe_traj is None:
                    loguru.logger.debug("created safe_traj!")
                    (self.safe_traj,) = ax.plot(T_pos[:, 0], T_pos[:, 1], color="C5", lw=1.2, ls="--", zorder=6.2)

                self.safe_traj.set_data(T_pos[:, 0], T_pos[:, 1])
            else:
                if self.opt_traj is None:
                    (self.opt_traj,) = ax.plot(T_pos[:, 0], T_pos[:, 1], color="C1", lw=1.3, zorder=6)

                self.opt_traj.set_data(T_pos[:, 0], T_pos[:, 1])
        else:
            bxT_state = np.array(trajectory_list)
            b2T_pos = bxT_state[:, :2, :]
            bT2_pos = ei.rearrange(b2T_pos, "b nx T -> b T nx")

            if self.traj_col is None:
                self.traj_col = LineCollection(bT2_pos, colors="C2", lw=0.2, alpha=0.4, zorder=5)
                ax.add_collection(self.traj_col)

            self.traj_col.set_segments(bT2_pos)

    @property
    def animated_artists(self):
        artists = [self.dubins_com, self.pcm, self.obs_col, self.title_text]
        if self.traj_col is not None:
            artists.append(self.traj_col)
        if self.opt_traj is not None:
            artists.append(self.opt_traj)
        if self.safe_traj is not None:
            artists.append(self.safe_traj)
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
