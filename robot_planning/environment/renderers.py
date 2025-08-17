from unittest.mock import right

import numpy as np
import ast

# from scipy.stats import rice_gen
# from typer.cli import state

from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import kinematics_factory_base
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from robot_planning.factory.factories import dynamics_factory_base

from robot_planning.helper.timer import Timer
from robot_planning.helper.utils import AUTORALLY_DYNAMICS_DIR
import os
import imageio
import moviepy.video.io.ImageSequenceClip
import einops as ei
from matplotlib.collections import LineCollection, PatchCollection
from robot_planning.helper.blit_manager import BlitManager
from timeit import default_timer
import time

class Renderer:
    def __init__(self):
        pass

    def initialize_from_config(self, config_data, section_name):
        pass

    def create_figure(self):
        pass

    def close_figure(self):
        pass

    def render_all(self):
        pass

    def render_agents(self):
        pass

    def render_states(self):
        pass

    def render_obstacles(self):
        pass

    def render_trajectories(self):
        pass

    def show(self):
        pass

    def clear(self):
        pass


class MatplotlibRenderer(Renderer):
    def __init__(
        self,
        xaxis_range=None,
        yaxis_range=None,
        auto_range=None,
        figure_size=None,
        figure_dpi=None,
        active=True,
        save_animation=False,
    ):
        Renderer.__init__(self)
        self.xaxis_range = xaxis_range
        self.yaxis_range = yaxis_range
        self.auto_range = auto_range
        self.figure_size = figure_size
        self.figure_dpi = figure_dpi
        self._figure = None
        self._axis: plt.Axes = None
        self.active = active
        self.save_animation = save_animation
        self.save_dir = None
        self.frame = 0

        self._has_paused = False
        self._bg = None

        self.show_every = None

    def initialize_from_config(self, config_data, section_name):
        Renderer.initialize_from_config(self, config_data, section_name)
        self.xaxis_range = np.asarray(
            ast.literal_eval(config_data.get(section_name, "xaxis_range")),
            dtype=np.float64,
        )
        self.yaxis_range = np.asarray(
            ast.literal_eval(config_data.get(section_name, "yaxis_range")),
            dtype=np.float64,
        )
        self.figure_size = np.asarray(
            ast.literal_eval(config_data.get(section_name, "figure_size")), dtype=int
        )
        self.figure_dpi = config_data.getint(section_name, "figure_dpi")
        if config_data.has_option(section_name, "save_dir"):
            self.save_dir = config_data.get(section_name, "save_dir")
        if config_data.has_option(section_name, "save_animation"):
            self.save_animation = config_data.getboolean(section_name, "save_animation")
        if config_data.has_option(section_name, "auto_range"):
            self.auto_range = ast.literal_eval(
                config_data.get(section_name, "auto_range")
            )
        else:
            self.auto_range = False
        if config_data.has_option(section_name, "equal_aspect_ratio"):
            self.equal_aspect_ratio = config_data.getboolean(section_name, "equal_aspect_ratio")
        else:
            self.equal_aspect_ratio = False
        self.create_figure()


    def set_save_dir(self, save_dir):
        assert save_dir is not None
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.save_dir = save_dir

    def create_figure(self):
        if self.active:
            self._figure = plt.figure(
                figsize=(self.figure_size[0], self.figure_size[1]), dpi=self.figure_dpi
            )
            self._axis = self._figure.add_subplot(1, 1, 1)
            plt.figure(self._figure.number)

            plt.show(block=False)
            # plt.pause(0.1)
            self._bg = self._figure.canvas.copy_from_bbox(self._figure.bbox)

    def close_figure(self):
        plt.close(self._figure)

    def set_range(self):
        if not self.auto_range:
            self._axis.axis(
                [
                    self.xaxis_range[0],
                    self.xaxis_range[1],
                    self.yaxis_range[0],
                    self.yaxis_range[1],
                ]
            )
        if self.equal_aspect_ratio is True:
            self._axis.set_aspect('equal')  # Keeps scaling identical
        plt.grid(True)

    def show(self):
        if not self._has_paused:
            self._has_paused = True

        if self.show_every is not None and self.frame % self.show_every != 0:
            self.frame += 1
            return

        timer = Timer.get_active()
        assert self.active
        timer_ = timer.child("set_range").start()
        self.set_range()
        timer_.stop()
        timer_ = timer.child("pause 1e-3").start()
        # plt.pause(0.01)
        # plt.pause(1e-3)
        self._figure.canvas.restore_region(self._bg)
        plt.draw()
        # for artist in self._axis.get_children():
        #     self._axis.draw_artist(artist)
        self._figure.canvas.blit(self._figure.bbox)
        self._figure.canvas.flush_events()

        timer_.stop()
        if self.save_animation:
            timer_ = timer.child("save").start()
            self.save()
            timer_.stop()
        self.frame += 1

    def clear(self):
        plt.cla()

    def save(self, save_path_name=None):
        assert self.active
        if save_path_name is None:
            assert self.save_dir is not None
            save_path_name = self.save_dir / "frame{}.png".format(self.frame)
        self.set_range()
        plt.savefig(save_path_name)

    def close(self):
        plt.close("all")

    def activate(self):
        self.active = True
        self.frame = 0

    def deactivate(self):
        self.close()
        self.active = False
        self.frame = 0

    def render_gif(self, duration=0.05):
        frames = [0 for _ in range(5)]
        for frame in range(self.frame):
            frames.append(frame)
        frames += [self.frame - 1 for _ in range(5)]

        images = []
        for frame in frames:
            file_name = self.save_dir / "frame{}.png".format(frame)
            images.append(imageio.imread(file_name))

        gif_dir = self.save_dir / "movie.gif"
        imageio.mimsave(gif_dir, images, duration=duration)

    def render_mp4(self, duration=0.5):
        image_folder = self.save_dir
        fps = int(1 / duration)

        frames = [0 for _ in range(5)]
        for frame in range(self.frame):
            frames.append(frame)
        frames += [self.frame - 1 for _ in range(5)]

        image_files = [
            os.path.join(image_folder, "frame{}.png".format(frame)) for frame in frames
        ]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
            image_files, fps=fps
        )
        clip.write_videofile(str(self.save_dir / "movie.mp4"))




class MPPIMatplotlibRenderer(MatplotlibRenderer):
    def __init__(
        self,
        xaxis_range=None,
        yaxis_range=None,
        auto_range=None,
        figure_size=None,
        figure_dpi=None,
        trajectories_rendering=True,
    ):
        MatplotlibRenderer.__init__(
            self, xaxis_range, yaxis_range, auto_range, figure_size, figure_dpi
        )
        self.trajectories_rendering = trajectories_rendering

    def initialize_from_config(self, config_data, section_name):
        MatplotlibRenderer.initialize_from_config(self, config_data, section_name)
        if config_data.has_option(section_name, "trajectories_rendering"):
            self.trajectories_rendering = config_data.getboolean(
                section_name, "trajectories_rendering"
            )

    def render_states(self, state_list=None, kinematics_list=None, **kwargs):
        if self.active:
            for i in range(len(state_list)):
                state = state_list[i]
                kinematics = kinematics_list[i]
                circle = plt.Circle((state[0], state[1]), kinematics.radius, **kwargs)
                self._axis.add_artist(circle)

    def render_obstacles(self, obstacle_list=None, **kwargs):
        if self.active:
            for (x, y, size) in obstacle_list:
                circle = plt.Circle((x, y), size, **kwargs)
                self._axis.add_artist(circle)

    def render_goal(self, goal=None, **kwargs):
        if self.active:
            x = goal[0]
            y = goal[1]
            radius = goal[2]
            circle = plt.Circle((x, y), radius, **kwargs)
            self._axis.add_artist(circle)

    def render_trajectories(self, trajectory_list=None, **kwargs):
        if self.active and self.trajectories_rendering:
            for trajectory in trajectory_list:
                previous_state = trajectory[:, 0]
                for i in range(1, trajectory.shape[1]):
                    state = trajectory[:, i]
                    (line,) = self._axis.plot(
                        [state[0], previous_state[0]],
                        [state[1], previous_state[1]],
                        **kwargs
                    )
                    previous_state = state


class EnvMatplotlibRenderer(MatplotlibRenderer):
    def __init__(
        self,
        xaxis_range=None,
        yaxis_range=None,
        auto_range=None,
        figure_size=None,
        figure_dpi=None,
        rendering_trajectory=True,
    ):
        MatplotlibRenderer.__init__(
            self, xaxis_range, yaxis_range, auto_range, figure_size, figure_dpi
        )
        self.n_agents = None
        self.isAgentsUpdated = None
        self.rendering_trajectory = rendering_trajectory

    def initialize_from_config(self, config_data, section_name):
        MatplotlibRenderer.initialize_from_config(self, config_data, section_name)

    def set_n_agents(self, n_agents):
        self.n_agents = n_agents
        self.isAgentsUpdated = [False for _ in range(self.n_agents)]

    def render_states(self, state_list=None, kinematics_list=None, **kwargs):
        if self.active:
            for i in range(len(state_list)):
                state = state_list[i]
                kinematics = kinematics_list[i]
                if hasattr(kinematics, "color"):
                    circle = plt.Circle(
                        (state[0], state[1]),
                        kinematics.radius,
                        color=kinematics.color,
                        **kwargs
                    )
                else:
                    circle = plt.Circle(
                        (state[0], state[1]), kinematics.radius, **kwargs
                    )
                self._axis.add_artist(circle)

    def render_obstacles(self, obstacle_list=None, **kwargs):
        if self.active:
            for (ox, oy, size) in obstacle_list:
                circle = plt.Circle((ox, oy), size, **kwargs)
                self._axis.add_artist(circle)

    def render_goal(self, goal=None, **kwargs):
        if self.active:
            x = goal[0]
            y = goal[1]
            radius = goal[2]
            circle = plt.Circle((x, y), radius, **kwargs)
            self._axis.add_artist(circle)

    def render_trajectories(self, trajectory_list=None, **kwargs):
        if self.active and self.rendering_trajectory:
            for trajectory in trajectory_list:
                previous_state = trajectory[:, 0]
                for i in range(1, trajectory.shape[1]):
                    state = trajectory[:, i]
                    (line,) = self._axis.plot(
                        [state[0], previous_state[0]],
                        [state[1], previous_state[1]],
                        **kwargs
                    )
                    previous_state = state

    def clear(self):
        if all(self.isAgentsUpdated):
            plt.cla()
            self.isAgentsUpdated = [False for _ in range(self.n_agents)]

    def show(self):
        assert self.active
        if all(self.isAgentsUpdated):
            self.set_range()
            # plt.pause(0.01)
            if self.save_animation:
                self.save()
            self.frame += 1


class CSSMPCMatplotlibRenderer(MatplotlibRenderer):
    def __init__(
        self,
        xaxis_range=None,
        yaxis_range=None,
        auto_range=None,
        figure_size=None,
        figure_dpi=None,
    ):
        MatplotlibRenderer.__init__(
            self, xaxis_range, yaxis_range, auto_range, figure_size, figure_dpi
        )

    def initialize_from_config(self, config_data, section_name):
        MatplotlibRenderer.initialize_from_config(self, config_data, section_name)

    def render_states(self, state_list=None, kinematics_list=None, **kwargs):
        for i in range(len(state_list)):
            state = state_list[i]
            kinematics = kinematics_list[i]
            circle = plt.Circle((state[-1], state[-2]), kinematics.radius, **kwargs)
            self._axis.add_artist(circle)

    def render_obstacles(self, obstacle_list=None, **kwargs):
        # for (ox, oy, size) in obstacle_list:
        #     circle = plt.Circle((ox, oy), size, **kwargs)
        #     self._axis.add_artist(circle)
        return

    def render_goal(self, goal=None, **kwargs):
        # x = goal[0]
        # y = goal[1]
        # radius = goal[2]
        # circle = plt.Circle((x, y), radius, **kwargs)
        # self._axis.add_artist(circle)
        return

    def render_trajectories(self, trajectory_list=None, **kwargs):
        for trajectory in trajectory_list:
            previous_state = trajectory[:, 0]
            for i in range(1, trajectory.shape[1]):
                state = trajectory[:, i]
                (line,) = self._axis.plot(
                    [state[-1], previous_state[-1]],
                    [state[-2], previous_state[-2]],
                    **kwargs
                )
                previous_state = state


class AutorallyMatplotlibRenderer2(MatplotlibRenderer):
    def __init__(
        self,
        xaxis_range=None,
        yaxis_range=None,
        auto_range=None,
        figure_size=None,
        figure_dpi=None,
        trajectories_rendering=True,
    ):
        MatplotlibRenderer.__init__(
            self, xaxis_range, yaxis_range, auto_range, figure_size, figure_dpi
        )
        self.trajectories_rendering = trajectories_rendering
        self.path_rendering = False
        self.path = np.zeros((3, 0))
        self.cbar = None

    def initialize_from_config(self, config_data, section_name):
        if config_data.has_option(section_name, "trajectories_rendering"):
            self.trajectories_rendering = config_data.getboolean(
                section_name, "trajectories_rendering"
            )
        MatplotlibRenderer.initialize_from_config(self, config_data, section_name)
        if config_data.has_option(section_name, "map_file"):
            map_file = config_data.get(section_name, "map_file")
            self.map = np.load(AUTORALLY_DYNAMICS_DIR + "/" + map_file)
        if config_data.has_option(section_name, "path_rendering"):
            self.path_rendering = config_data.get(section_name, "path_rendering")

    def render_states(self, state_list=None, kinematics_list=None, **kwargs):
        for i in range(len(state_list)):
            state = state_list[i]
            kinematics = kinematics_list[i]
            circle = plt.Circle((state[-2], state[-1]), kinematics.radius, **kwargs)
            self._axis.add_artist(circle)
        if self.path_rendering:
            self.path = np.append(
                self.path, np.vstack((state[0], state[6], state[7])), axis=1
            )
            pcm = self._axis.scatter(
                self.path[1, :], self.path[2, :], c=self.path[0, :], marker="."
            )
            if self.path.shape[1] < 2:
                self.cbar = plt.colorbar(pcm)
                self.cbar.set_label("speed (m/s)")
            else:
                self.cbar.update_normal(pcm)

    def render_obstacles(self, obstacle_list=None, **kwargs):
        self._axis.plot(self.map["X_in"], self.map["Y_in"], "k")
        self._axis.plot(self.map["X_out"], self.map["Y_out"], "k")
        for (x, y, size) in obstacle_list:
            circle = plt.Circle((x, y), size, **kwargs)
            self._axis.add_artist(circle)
        return

    def render_goal(self, goal=None, **kwargs):
        x = goal[0]
        y = goal[1]
        radius = goal[2]
        circle = plt.Circle((x, y), radius, **kwargs, zorder=0)
        self._axis.add_artist(circle)
        return

    def render_trajectories(self, trajectory_list=None, **kwargs):
        if self.trajectories_rendering is True:
            trajectory_list = np.asarray(trajectory_list)
            previous_state = trajectory_list[:, :, 0]
            for i in range(1, trajectory_list.shape[2]):
                state = trajectory_list[:, :, i]
                self._axis.plot(
                    [state[:, -2], previous_state[:, -2]],
                    [state[:, -1], previous_state[:, -1]],
                    **kwargs
                )
                previous_state = state
        else:
            pass


class CLBFAutorallyMatplotlibRenderer(MatplotlibRenderer):
    def __init__(
        self,
        xaxis_range=None,
        yaxis_range=None,
        auto_range=None,
        figure_size=None,
        figure_dpi=None,
        trajectories_rendering=True,
    ):
        MatplotlibRenderer.__init__(
            self, xaxis_range, yaxis_range, auto_range, figure_size, figure_dpi
        )
        self.trajectories_rendering = trajectories_rendering
        self.path_rendering = False
        self.path = np.zeros((3, 0))
        self.cbar = None

    def initialize_from_config(self, config_data, section_name):
        if config_data.has_option(section_name, "trajectories_rendering"):
            self.trajectories_rendering = config_data.getboolean(
                section_name, "trajectories_rendering"
            )
        MatplotlibRenderer.initialize_from_config(self, config_data, section_name)
        if config_data.has_option(section_name, "map_file"):
            map_file = config_data.get(section_name, "map_file")
            self.map = np.load(AUTORALLY_DYNAMICS_DIR + "/" + map_file)
        if config_data.has_option(section_name, "path_rendering"):
            self.path_rendering = config_data.get(section_name, "path_rendering")

    def render_states(self, state_list=None, kinematics_list=None, **kwargs):
        for i in range(len(state_list)):
            state = state_list[i]
            kinematics = kinematics_list[i]
            circle = plt.Circle((state[-2], state[-1]), kinematics.radius, **kwargs)
            self._axis.add_artist(circle)
        if self.path_rendering:
            # Path contains speed, x, and y
            self.path = np.append(
                self.path, np.vstack((state[2], state[-2], state[-1])), axis=1
            )
            pcm = self._axis.scatter(
                self.path[1, :], self.path[2, :], c=self.path[0, :], marker="."
            )
            if self.path.shape[1] < 2:
                self.cbar = plt.colorbar(pcm)
                self.cbar.set_label("speed (m/s)")
            else:
                self.cbar.update_normal(pcm)

    def render_obstacles(self, obstacle_list=None, **kwargs):
        self._axis.plot(self.map["X_in"], self.map["Y_in"], "k")
        self._axis.plot(self.map["X_out"], self.map["Y_out"], "k")
        return

    def render_goal(self, goal=None, **kwargs):
        x = goal[0]
        y = goal[1]
        radius = goal[2]
        circle = plt.Circle((x, y), radius, **kwargs, zorder=0)
        self._axis.add_artist(circle)
        return

    def render_trajectories(self, trajectory_list=None, **kwargs):
        trajectory_list = np.asarray(trajectory_list)
        if self.trajectories_rendering is True:
            previous_state = trajectory_list[:, :, 0]
            for i in range(1, trajectory_list.shape[2]):
                state = trajectory_list[:, :, i]
                self._axis.plot(
                    [state[:, -2], previous_state[:, -2]],
                    [state[:, -1], previous_state[:, -1]],
                    **kwargs
                )
                previous_state = state
        else:
            pass

class Quadrotor2DMatplotlibRenderer(MatplotlibRenderer):
    def __init__(
        self,
        xaxis_range=None,
        yaxis_range=None,
        auto_range=None,
        figure_size=None,
        figure_dpi=None,
        trajectories_rendering=True,
    ):
        MatplotlibRenderer.__init__(
            self, xaxis_range, yaxis_range, auto_range, figure_size, figure_dpi
        )

    def initialize_from_config(self, config_data, section_name):
        MatplotlibRenderer.initialize_from_config(self, config_data, section_name)
        dynamics_section_name = config_data.get(section_name, "dynamics")
        self.dynamics = factory_from_config(
            dynamics_factory_base, config_data, dynamics_section_name
        )
        self.figure_size = np.asarray(ast.literal_eval(config_data.get(section_name, "figure_size")), dtype=int)
        self.figure_dpi = config_data.getint(section_name, "figure_dpi")
        if config_data.has_option(section_name, "grid_on"):
            self.grid_on = config_data.getboolean(section_name, "grid_on")
        else:
            self.grid_on = True
        self._axis.grid(self.grid_on) #TODO: grid cannot be turned off

    def render_states(self, state_list=None, kinematics_list=None, **kwargs):
        for i in range(len(state_list)):
            state = state_list[i]
            kinematics = kinematics_list[i]
            self.render_quadrotor_2d(state, kinematics, **kwargs)

    def render_goal(self, goal=None, **kwargs):
        # self.render_quadrotor_2d(goal, self.dynamics, **kwargs)
        assert goal is not None
        goal_x = goal[0]
        self._axis.axvline(goal_x, color="green")


    def render_quadrotor_2d(self, state, kinematics, **kwargs):
        half_body_length_to_rotor_size_ratio = 2
        x = state[0]
        z = state[1]
        theta = state[2]
        left_rotor_pos = (x - kinematics.l * np.cos(theta), z - kinematics.l * np.sin(theta))
        circle_left_rotor = plt.Circle(left_rotor_pos, kinematics.l/half_body_length_to_rotor_size_ratio, **kwargs)
        self._axis.add_artist(circle_left_rotor)
        right_rotor_pos = np.array([x + kinematics.l * np.cos(theta), z + kinematics.l * np.sin(theta)])
        circle_right_rotor = plt.Circle(right_rotor_pos, kinematics.l/half_body_length_to_rotor_size_ratio, **kwargs)
        self._axis.add_artist(circle_right_rotor)
        circleCOM = plt.Circle((x, z), kinematics.l/half_body_length_to_rotor_size_ratio, **kwargs)
        self._axis.add_artist(circleCOM)
        #Add line connecting the rotors
        QuadrotorBodyLine = Line2D([left_rotor_pos[0], right_rotor_pos[0]], [left_rotor_pos[1], right_rotor_pos[1]], **kwargs)
        self._axis.add_artist(QuadrotorBodyLine)

        collision_circle_radius = kinematics.get_radius()
        circ = plt.Circle((x, z), collision_circle_radius, facecolor="blue", alpha=0.5, edgecolor="none")
        self._axis.add_artist(circ)

    def render_obstacles(self, obstacle_list=None, **kwargs):
        import matplotlib.patches as patches
        for (x, y, size) in obstacle_list:
            circle = plt.Circle((x, y), size, **kwargs)
            self._axis.add_artist(circle)
        rect = patches.Rectangle((-10, -5), 20, 4.95, facecolor='black')
        self._axis.add_artist(rect)
        return

    # def _create_figure(self) -> tuple[plt.Figure, plt.Axes]:
    #     fig = plt.figure(figsize=(self.figure_size[0], self.figure_size[1]), dpi=self.figure_dpi)
    #     ax = fig.add_subplot(1, 1, 1)
    #     ax.set_aspect("equal")
    #     return fig, ax

    def render_trajectories(self, trajectory_list=None, **kwargs):
        pass
        # if not self.trajectories_rendering:
        #     return
        # fig, ax = self._get_figax()
        # is_opt_traj = len(trajectory_list) == 1
        # if is_opt_traj:
        #     xT_state = np.array(trajectory_list[0])
        #     T_pos = xT_state[:2, :].T
        #     if self.opt_traj is None:
        #         (self.opt_traj,) = ax.plot(T_pos[:, 0], T_pos[:, 1], color="C1", lw=1.2, zorder=6)
        #     self.opt_traj.set_data(T_pos[:, 0], T_pos[:, 1])
        # else:
        #     bxT_state = np.array(trajectory_list)
        #     b2T_pos = bxT_state[:, :2, :]
        #     bT2_pos = ei.rearrange(b2T_pos, "b nx T -> b T nx")
        #     if self.traj_col is None:
        #         self.traj_col = LineCollection(bT2_pos, colors="C2", lw=0.2, alpha=0.4, zorder=5)
        #         ax.add_collection(self.traj_col)
        #     self.traj_col.set_segments(bT2_pos)


class DubinsMatplotlibRenderer(MatplotlibRenderer):
    def __init__(
            self,
            xaxis_range=None,
            yaxis_range=None,
            auto_range=None,
            figure_size=None,
            figure_dpi=None,
            trajectories_rendering=True,
    ):
        MatplotlibRenderer.__init__(
            self, xaxis_range, yaxis_range, auto_range, figure_size, figure_dpi
        )

    def initialize_from_config(self, config_data, section_name):
        MatplotlibRenderer.initialize_from_config(self, config_data, section_name)
        dynamics_section_name = config_data.get(section_name, "dynamics")
        self.dynamics = factory_from_config(
            dynamics_factory_base, config_data, dynamics_section_name
        )
        self.figure_size = np.asarray(ast.literal_eval(config_data.get(section_name, "figure_size")), dtype=int)
        self.figure_dpi = config_data.getint(section_name, "figure_dpi")
        if config_data.has_option(section_name, "grid_on"):
            self.grid_on = config_data.getboolean(section_name, "grid_on")
        else:
            self.grid_on = True
        self._axis.grid(self.grid_on) #TODO: grid cannot be turned off

    def render_states(self, state_list=None, kinematics_list=None, **kwargs):
        for i in range(len(state_list)):
            state = state_list[i]
            kinematics = kinematics_list[i]
            self.render_dubins(state, kinematics, **kwargs)

    def render_goal(self, goal=None, **kwargs):
        # self.render_quadrotor_2d(goal, self.dynamics, **kwargs)
        assert goal is not None
        circ = plt.Circle((goal[0], goal[1]), goal[2], facecolor="green", alpha=0.5, edgecolor="none")
        self._axis.add_artist(circ)


    def render_dubins(self, state, kinematics, **kwargs):
        half_body_length_to_rotor_size_ratio = 2
        x = state[0]
        y = state[1]
        theta = state[2]

        collision_circle_radius = kinematics.get_radius()
        circ = plt.Circle((x, y), collision_circle_radius, facecolor="blue", alpha=0.5, edgecolor="none")
        self._axis.add_artist(circ)

    def render_obstacles(self, obstacle_list=None, **kwargs):
        for (x, y, size) in obstacle_list:
            circle = plt.Circle((x, y), size, **kwargs)
            self._axis.add_artist(circle)
        return

    # def _create_figure(self) -> tuple[plt.Figure, plt.Axes]:
    #     fig = plt.figure(figsize=(self.figure_size[0], self.figure_size[1]), dpi=self.figure_dpi)
    #     ax = fig.add_subplot(1, 1, 1)
    #     ax.set_aspect("equal")
    #     return fig, ax

    def render_trajectories(self, trajectory_list=None, **kwargs):
        for trajectory in trajectory_list:
            previous_state = trajectory[:, 0]
            for i in range(1, trajectory.shape[1]):
                state = trajectory[:, i]
                (line,) = self._axis.plot(
                    [state[0], previous_state[0]],
                    [state[1], previous_state[1]],
                    **kwargs
                )
                previous_state = state



