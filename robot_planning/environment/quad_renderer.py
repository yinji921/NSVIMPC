import ast
import loguru
import time
from timeit import default_timer

import einops as ei
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import ColormapRegistry
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text

from robot_planning.helper.blit_manager import BlitManager
from robot_planning.helper.convenience import get_drone_kinematics
from robot_planning.environment.dynamics.autorally_dynamics.map_coords import MapCA
from robot_planning.environment.mpl_renderer import MutablePatchCollection, register_cmaps
from robot_planning.environment.renderers import Renderer
from robot_planning.factory.factories import dynamics_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.helper.utils import AUTORALLY_DYNAMICS_DIR


class QuadrotorMatplotlibRendererBlit(Renderer):
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

        # The quadrotor consists of circles and a line.
        self.rotor_L: plt.Circle | None = None
        self.rotor_R: plt.Circle | None = None
        self.quad_com: plt.Circle | None = None
        self.quad_line: plt.Line2D | None = None
        self.quad_col: plt.Circle | None = None

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

        self.kin = get_drone_kinematics()

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

    def set_quadrotor_artists(self, state: np.ndarray):
        kinematics = self.kin

        x, z, theta = state[:3]
        half_body_length_to_rotor_size_ratio = 2

        rotor_radius = kinematics.l / half_body_length_to_rotor_size_ratio

        left_rotor_pos = (x - kinematics.l * np.cos(theta), z - kinematics.l * np.sin(theta))
        self.rotor_L.set_center(left_rotor_pos)
        if self.rotor_L.radius != rotor_radius:
            self.rotor_L.set_radius(rotor_radius)

        right_rotor_pos = np.array([x + kinematics.l * np.cos(theta), z + kinematics.l * np.sin(theta)])
        self.rotor_R.set_center(right_rotor_pos)
        if self.rotor_R.radius != rotor_radius:
            self.rotor_R.set_radius(rotor_radius)

        com_pos = np.array([x, z])
        self.quad_com.set_center(com_pos)
        if self.quad_com.radius != kinematics.l / half_body_length_to_rotor_size_ratio:
            self.quad_com.set_radius(kinematics.l / half_body_length_to_rotor_size_ratio)

        # Add line connecting the rotors
        self.quad_line.set_data([left_rotor_pos[0], right_rotor_pos[0]], [left_rotor_pos[1], right_rotor_pos[1]])

        collision_circle_radius = kinematics.get_radius()
        self.quad_col.set_center(com_pos)
        if self.quad_col.radius != collision_circle_radius:
            self.quad_col.set_radius(collision_circle_radius)

    def create_quadrotor_artists(self):
        assert self.rotor_L is None
        kinematics = self.kin

        half_body_length_to_rotor_size_ratio = 2
        rotor_radius = kinematics.l / half_body_length_to_rotor_size_ratio

        color_rotor_L = "blue"
        color_rotor_R = "red"
        color_com = "black"
        col_style = dict(facecolor="blue", edgecolor="none", alpha=0.2)

        self.rotor_L = plt.Circle((0, 0), rotor_radius, facecolor=color_rotor_L, edgecolor="none", zorder=6)
        self.rotor_R = plt.Circle((0, 0), rotor_radius, facecolor=color_rotor_R, edgecolor="none", zorder=6)
        self.quad_com = plt.Circle((0, 0), rotor_radius, facecolor=color_com, edgecolor="none", zorder=5.5)
        self.quad_line = Line2D([0, 0], [0, 0], color="black", lw=1.0, zorder=5.2)
        self.quad_col = plt.Circle((0, 0), rotor_radius, zorder=5.1, **col_style)

        self.title_text = self._ax.set_title("")

        self._ax.add_artist(self.rotor_L)
        self._ax.add_artist(self.rotor_R)
        self._ax.add_artist(self.quad_com)
        self._ax.add_artist(self.quad_line)
        self._ax.add_artist(self.quad_col)

    def render_states(self, state_list=None, kinematics_list=None, **kwargs):
        if not self.active:
            return

        fig, ax = self._get_figax()

        assert len(state_list) == len(kinematics_list) == 1
        state, kinematics = state_list[0], kinematics_list[0]

        if self.rotor_L is None:
            self.create_quadrotor_artists()

        self.set_quadrotor_artists(state)

        # Update the current title with the velocity info.
        vel = np.linalg.norm(state[3:5])
        text = r"vel: [ ${:+.1f}$, ${:+.1f}$ ]   ( ${:+.1f}$".format(state[3], state[4], vel)
        self.title_text.set_text(text)

        if not self.path_rendering:
            return

        # [vel, px, pz]
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
            self.cbar.set_label("speed (m/s)")

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
        pass
        # if self.goal is not None:
        #     return
        # fig, ax = self._get_figax()
        # x = goal[0]
        # y = goal[1]
        # radius = goal[2]
        # self.goal = plt.Circle((x, y), radius, **kwargs, zorder=0)
        # ax.add_patch(self.goal)

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
        artists = [self.rotor_L, self.rotor_R, self.quad_com, self.quad_col, self.pcm, self.obs_col, self.title_text]
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
