import copy
import pickle
from collections import defaultdict

import numpy as onp
import jax
import csv
import jax.numpy as np
import os

from robot_planning.environment.dynamics.autorally_dynamics.map_coords import MapCA
from robot_planning.helper.timer import Timer
from robot_planning.helper.utils import EXPERIMENT_ROOT_DIR
from robot_planning.factory.factories import (
    collision_checker_factory_base,
    goal_checker_factory_base,
)
from robot_planning.factory.factory_from_config import factory_from_config


class Logger(object):
    def __init__(self, experiment_root_dir=None, experiment_name=None):
        self.experiment_root_dir = None
        self.experiment_name = None
        self.experiments_folder_name = None

    def initialize_from_config(self, config_data, section_name):
        self.experiment_root_dir = EXPERIMENT_ROOT_DIR
        if config_data.has_option(section_name, "experiment_name"):
            self.experiment_name = config_data.get(section_name, "experiment_name")

    def save_fig(self):
        raise NotImplementedError

    def set_experiment_name(self, experiment_name):
        self.experiment_name = experiment_name

    def create_save_dir(self):
        experiments_dir = self.experiment_root_dir + "/" + self.experiments_folder_name
        if not os.path.isdir(experiments_dir):
            os.mkdir(experiments_dir)
        current_experiment_dir = (
            self.experiment_root_dir
            + "/"
            + self.experiments_folder_name
            + "/"
            + self.experiment_name
        )
        if not os.path.isdir(current_experiment_dir):
            os.mkdir(current_experiment_dir)
        return experiments_dir, current_experiment_dir

    def shutdown(self):
        return


class MPPILogger(Logger):
    def __init__(self, experiment_dir=None):
        Logger.__init__(self, experiment_dir)
        self.experiments_folder_name = "MPPI_experiments"

    def initialize_from_config(self, config_data, section_name):
        Logger.initialize_from_config(self, config_data, section_name)

    def save_fig(self, renderer=None, time=None):
        self.create_save_dir()
        time = str(np.around(time, decimals=2))
        save_path_name = (
            self.experiment_root_dir
            + "/"
            + self.experiments_folder_name
            + "/"
            + self.experiment_name
            + "/"
            + time
            + ".png"
        )
        renderer.save(save_path_name)


class AutorallyLogger(Logger):
    def __init__(self, experiment_dir=None, collision_checker=None, goal_checker=None):
        Logger.__init__(self, experiment_dir)
        self.experiments_folder_name = "Autorally_experiments"
        self.number_of_collisions = 0
        self.number_of_laps = 0
        self.number_of_failure = 0
        self.in_obstacle = False
        self.around_goal_position = True
        self.collision_checker = collision_checker
        self.goal_checker = goal_checker

    def initialize_from_config(self, config_data, section_name):
        Logger.initialize_from_config(self, config_data, section_name)
        if config_data.has_option(section_name, "goal_checker"):
            goal_checker_section_name = config_data.get(section_name, "goal_checker")
            self.goal_checker = factory_from_config(
                goal_checker_factory_base, config_data, goal_checker_section_name
            )
        if config_data.has_option(section_name, "collision_checker"):
            collision_checker_section_name = config_data.get(
                section_name, "collision_checker"
            )
            self.collision_checker = factory_from_config(
                collision_checker_factory_base,
                config_data,
                collision_checker_section_name,
            )
        _, current_experiment_dir = self.create_save_dir()
        self.log_file_path = (
            current_experiment_dir + "/" + self.experiment_name + "_log_file.csv"
        )
        with open(self.log_file_path, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                [
                    "robot time",
                    "state",
                    "lap_num",
                    "collision_num",
                    "controller_failure_num",
                ]
            )

    def set_agent(self, agent):
        self.agent = agent
        # self.in_obstacle = self.collision_checker.check(agent.get_state())
        self.in_obstacle = False
        self.around_goal_position = self.goal_checker.check(agent.get_state())

    def calculate_number_of_collisions(self, state, dynamics, collision_checker):
        # This state is in cartesian coordinates, needs to be converted to map coordinates
        state = self.global_to_local_coordinate_transform(state, dynamics)
        if (
            collision_checker.check(state) and self.in_obstacle is False
        ):  # the limit should not be hard-coded
            self.in_obstacle = True
            self.number_of_collisions += 1
        if not collision_checker.check(state) and self.in_obstacle is True:
            self.in_obstacle = False

    def calculate_number_of_laps(self, state, dynamics, goal_checker):
        # This state is in cartesian coordinates, needs to be converted to map coordinates
        # state = self.global_to_local_coordinate_transform(state, dynamics)
        if (
            goal_checker.check(state) and self.around_goal_position is False
        ):  # the limit should not be hard-coded
            self.around_goal_position = True
            self.number_of_laps += 1
        if not goal_checker.check(state) and self.around_goal_position is True:
            self.around_goal_position = False

    def add_number_of_failure(self):
        self.number_of_failure += 1

    def log(self):
        info = [
            self.agent.get_time(),
            self.agent.get_state(),
            self.number_of_laps,
            self.number_of_collisions,
            self.number_of_failure,
        ]
        with open(self.log_file_path, "a") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(info)

    def global_to_local_coordinate_transform(self, state, dynamics):
        state = copy.deepcopy(state)
        e_psi, e_y, s = dynamics.track.localize(
            np.array((state[-2], state[-1])), state[-3]
        )
        state = state.at[-3:].set(np.vstack((e_psi, e_y, s)).reshape((3,)))
        return state

    def get_num_of_collisions(self):
        return copy.deepcopy(self.number_of_collisions)

    def get_num_of_laps(self):
        return copy.deepcopy(self.number_of_laps)

    def get_num_of_failures(self):
        return copy.deepcopy(self.number_of_failure)


class AutorallyNpzLogger(Logger):
    def __init__(self, experiment_dir=None, collision_checker=None, goal_checker=None):
        Logger.__init__(self, experiment_dir)
        self.experiments_folder_name = "Autorally_experiments"
        self.number_of_collisions = 0
        self.number_of_laps = 0
        self.number_of_failure = 0
        self.in_obstacle = False
        self.around_goal_position = True
        self.collision_checker = collision_checker
        self.goal_checker = goal_checker
        # self.sim_states = np.zeros((11, 0))
        self.sim_states_list = []
        # self.disturbances = np.zeros((8, 0))
        self.disturbances_list = []
        self.crash = 0

        # Eric
        self.timeouts = 0
        self.omegas = []  # weights in ESS, i.e. the w_i in \sum^N_{i=1} w_i \cdot \mu_i
        self.ctrl_hz = 0
        self.misc = defaultdict(list)

        self.localize_jit = None
        self.vmap_localize_jit = None

    def initialize_from_config(self, config_data, section_name):
        Logger.initialize_from_config(self, config_data, section_name)
        if config_data.has_option(section_name, "goal_checker"):
            goal_checker_section_name = config_data.get(section_name, "goal_checker")
            self.goal_checker = factory_from_config(
                goal_checker_factory_base, config_data, goal_checker_section_name
            )
        if config_data.has_option(section_name, "collision_checker"):
            collision_checker_section_name = config_data.get(
                section_name, "collision_checker"
            )
            self.collision_checker = factory_from_config(
                collision_checker_factory_base,
                config_data,
                collision_checker_section_name,
            )
        self.batch_code = config_data.get(section_name, "batch_code")
        _, current_experiment_dir = self.create_save_dir()
        self.log_file_path = (
            current_experiment_dir + "/" + self.experiment_name + "_log"
        )

        if config_data.has_option(section_name, "num_states"):
            # self.sim_states = np.zeros(
            #     (config_data.getint(section_name, "num_states"), 0)
            # )
            pass

    def create_save_dir(self):
        experiments_dir = self.experiment_root_dir + "/" + self.experiments_folder_name
        if not os.path.isdir(experiments_dir):
            os.mkdir(experiments_dir)
        current_experiment_dir = (
            self.experiment_root_dir
            + "/"
            + self.experiments_folder_name
            + "/"
            + self.batch_code
        )
        if not os.path.isdir(current_experiment_dir):
            os.mkdir(current_experiment_dir)
        return experiments_dir, current_experiment_dir

    def set_agent(self, agent):
        self.agent = agent
        self.in_obstacle = False
        # self.in_obstacle = self.collision_checker.check(agent.get_state())
        self.around_goal_position = self.goal_checker.check(agent.get_state())

        # self.localize_jit = jax.jit(self.agent.dynamics.track.localize)
        track: MapCA = self.agent.dynamics.track
        self.localize_jit = jax.jit(track.localize_lerp)
        self.vmap_localize_jit = jax.jit(jax.vmap(track.localize_lerp))

    def calculate_number_of_collisions(self, state, dynamics, collision_checker):
        # This state is in cartesian coordinates, needs to be converted to map coordinates
        local_state = self.global_to_local_coordinate_transform(state)
        if (
            collision_checker.check(local_state, state) and self.in_obstacle is False
        ):  # the limit should not be hard-coded
            self.in_obstacle = True
            self.number_of_collisions += 1
        if not collision_checker.check(local_state, state) and self.in_obstacle is True:
            self.in_obstacle = False

    def increment_timeouts(self):
        self.timeouts += 1

    def add_omega(self, omega):
        self.omegas.append(omega.squeeze())  # Expected shape of omega is (N,) for N num_trajectories

    def set_ctrl_hz(self, ctrl_hz):
        self.ctrl_hz = ctrl_hz

    def calculate_number_of_failures(self, state, dynamics, collision_checker):
        # This state is in cartesian coordinates, needs to be converted to map coordinates
        local_state = self.global_to_local_coordinate_transform(state)
        if collision_checker.check(local_state, state):
            self.number_of_failure = 1

    def calculate_number_of_laps(self, state, dynamics, goal_checker):
        # This state is in cartesian coordinates, needs to be converted to map coordinates
        # state = self.global_to_local_coordinate_transform(state, dynamics)
        if (
            goal_checker.check(state) and self.around_goal_position is False
        ):  # the limit should not be hard-coded
            self.around_goal_position = True
            self.number_of_laps += 1
        if not goal_checker.check(state) and self.around_goal_position is True:
            self.around_goal_position = False

    def calculate_agent_disturbance(self, state_next, old_state, action, nominal_dynamics):
        old_state = old_state.reshape((-1, 1))
        nominal_state = nominal_dynamics.propagate(old_state.reshape(-1, 1), action.reshape(-1, 1)).reshape(old_state.shape)
        disturbance = state_next.reshape((-1, 1)) - nominal_state
        self.disturbances_list.append(disturbance)
        # self.disturbances = np.append(
        #     self.disturbances,
        #     disturbance,
        #     axis=1,
        # )


    def add_number_of_failure(self):
        self.number_of_failure += 1

    def log(self):
        timer = Timer.get_active()
        if self.number_of_failure > 0:
            self.crash = 1
        state = self.agent.get_state()
        timer_ = timer.child("global_to_local_coordinate_transform").start()
        map_coords = self.global_to_local_coordinate_transform(state)[-3:]
        timer_.stop()
        timer_ = timer.child("append").start()

        sim_state = onp.vstack((onp.array(state.reshape((-1, 1))), onp.array(map_coords.reshape((-1, 1)))))
        self.sim_states_list.append(sim_state)
        # self.sim_states = np.append(
        #     self.sim_states,
        #     np.vstack((state.reshape((-1, 1)), map_coords.reshape((-1, 1)))),
        #     axis=1,
        # )
        timer_.stop()
        timer.stop().print_results()

    def global_to_local_coordinate_transform(self, state):
        state = state.copy().reshape((-1, 1))
        # e_psi, e_y, s = self.agent.dynamics.track.localize(
        #     np.array((state[-2, :], state[-1, :])), state[-3, :]
        # )

        # (nx, b)
        assert state.ndim == 2
        xb_state = state
        nx, b = xb_state.shape
        b_pos = xb_state[-2:, :].T
        b_psi = xb_state[-3, :]
        b_epsi, b_ey, b_s = self.vmap_localize_jit(b_pos, b_psi)

        # (3, b)
        xb_curvilinear = np.stack([b_epsi, b_ey, b_s], axis=0)
        assert xb_curvilinear.shape == (3, b)

        state = state.at[-3:, :].set(xb_curvilinear)

        # e_psi, e_y, s = self.localize_jit(np.array((state[-2, :], state[-1, :])), state[-3, :])
        # state = state.at[-3:, :].set(np.vstack((e_psi, e_y, s)).reshape((3, 1)))
        return state

    def get_num_of_collisions(self):
        return copy.deepcopy(self.number_of_collisions)

    def get_num_of_laps(self):
        return copy.deepcopy(self.number_of_laps)

    def get_num_of_failures(self):
        return copy.deepcopy(self.number_of_failure)

    @property
    def sim_states(self):
        return onp.concatenate(self.sim_states_list, axis=1)

    @property
    def n_sim_states(self):
        return len(self.sim_states_list)

    def shutdown(self, mean_eval_time=-1.0):
        disturbances = np.concatenate(self.disturbances_list, axis=1)
        self.disturbances = disturbances

        mean_velocity = self.sim_states[0, :].mean()
        max_velocity = self.sim_states[0, :].max()
        print(f"Logging to to {self.log_file_path}")
        np.savez(
            self.log_file_path,
            states=self.sim_states,
            collisions=self.number_of_collisions,
            crash=self.crash,
            number_of_laps=self.number_of_laps,
            lap_time=self.agent.get_time(),
            mean_velocity=mean_velocity,
            max_velocity=max_velocity,
            mean_eval_time=mean_eval_time,
            disturbances=self.disturbances,
            timeouts=self.timeouts,
            omegas=self.omegas,
            ctrl_hz=self.ctrl_hz,
        )

class Drone2DNpzLogger(Logger):
    def __init__(self, experiment_dir=None, collision_checker=None, goal_checker=None):
        Logger.__init__(self, experiment_dir)
        self.experiments_folder_name = "Drone2D_experiments"
        self.number_of_collisions = 0
        self.number_of_failure = 0
        self.in_obstacle = False
        self.around_goal_position = True
        self.collision_checker = collision_checker
        self.goal_checker = goal_checker
        # self.sim_states = np.zeros((11, 0))
        self.sim_states_list = []
        # self.disturbances = np.zeros((8, 0))
        self.disturbances_list = []
        self.crash = 0

        # Eric
        self.timeouts = 0
        self.omegas = []  # weights in ESS, i.e. the w_i in \sum^N_{i=1} w_i \cdot \mu_i
        self.ctrl_hz = 0
        self.trajectories_list = []
        self.opt_trajs = []

        self.localize_jit = None
        self.vmap_localize_jit = None

    def initialize_from_config(self, config_data, section_name):
        Logger.initialize_from_config(self, config_data, section_name)
        if config_data.has_option(section_name, "goal_checker"):
            goal_checker_section_name = config_data.get(section_name, "goal_checker")
            self.goal_checker = factory_from_config(
                goal_checker_factory_base, config_data, goal_checker_section_name
            )
        if config_data.has_option(section_name, "collision_checker"):
            collision_checker_section_name = config_data.get(
                section_name, "collision_checker"
            )
            self.collision_checker = factory_from_config(
                collision_checker_factory_base,
                config_data,
                collision_checker_section_name,
            )
        self.batch_code = config_data.get(section_name, "batch_code")
        _, current_experiment_dir = self.create_save_dir()
        self.log_file_path = (
            current_experiment_dir + "/" + self.experiment_name + "_log"
        )

        if config_data.has_option(section_name, "num_states"):
            pass

    def create_save_dir(self):
        experiments_dir = self.experiment_root_dir + "/" + self.experiments_folder_name
        if not os.path.isdir(experiments_dir):
            os.mkdir(experiments_dir)
        current_experiment_dir = (
            self.experiment_root_dir
            + "/"
            + self.experiments_folder_name
            + "/"
            + self.batch_code
        )
        if not os.path.isdir(current_experiment_dir):
            os.mkdir(current_experiment_dir)
        return experiments_dir, current_experiment_dir

    def set_agent(self, agent):
        self.agent = agent

    def increment_timeouts(self):
        self.timeouts += 1

    def add_omega(self, omega):
        self.omegas.append(omega.squeeze())  # Expected shape of omega is (N,) for N num_trajectories

    def set_ctrl_hz(self, ctrl_hz):
        self.ctrl_hz = ctrl_hz

    def calculate_number_of_failures(self, state, collision_checker):
        if collision_checker.check(state):
            self.number_of_failure = 1

    def add_number_of_failure(self):
        self.number_of_failure += 1

    def append_state(self):
        state = self.agent.get_state()
        self.sim_states_list.append(state.reshape((-1, 1)))

    def log(self):
        timer = Timer.get_active()
        if self.number_of_failure > 0:
            self.crash = 1
        state = self.agent.get_state()
        self.sim_states_list.append(state.reshape((-1, 1)))
        timer.stop().print_results()

    def add_trajectory_list(self, trajectory_list):
        self.trajectories_list.append(trajectory_list)


    def get_num_of_collisions(self):
        return copy.deepcopy(self.number_of_collisions)

    def get_num_of_failures(self):
        return copy.deepcopy(self.number_of_failure)

    @property
    def sim_states(self):
        return onp.concatenate(self.sim_states_list, axis=1)

    @property
    def n_sim_states(self):
        return len(self.sim_states_list)

    def shutdown(self, mean_eval_time=-1.0):
        # disturbances = np.concatenate(self.disturbances_list, axis=1)
        # self.disturbances = disturbances

        mean_velocity = self.sim_states[0, :].mean()
        max_velocity = self.sim_states[0, :].max()
        print(f"Logging to to {self.log_file_path}")
        np.savez(
            self.log_file_path,
            states=self.sim_states,
            collisions=self.number_of_collisions,
            crash=self.crash,
            lap_time=self.agent.get_time(),
            mean_velocity=mean_velocity,
            max_velocity=max_velocity,
            mean_eval_time=mean_eval_time,
            # disturbances=self.disturbances,
            timeouts=self.timeouts,
            omegas=self.omegas,
            ctrl_hz=self.ctrl_hz,
        )


class AutorallyMPPILogger(AutorallyLogger):
    def __init__(self, experiment_dir=None, collision_checker=None, goal_checker=None):
        AutorallyLogger.__init__(self, experiment_dir)
        self.experiments_folder_name = "Autorally_MPPI_experiments"

    def initialize_from_config(self, config_data, section_name):
        AutorallyLogger.initialize_from_config(self, config_data, section_name)


class Drone3DNpzLogger(Logger):
    def __init__(self, experiment_dir=None, collision_checker=None, goal_checker=None):
        Logger.__init__(self, experiment_dir)
        self.experiments_folder_name = "Drone3D_experiments"
        self.number_of_collisions = 0
        self.number_of_failure = 0
        self.in_obstacle = False
        self.around_goal_position = True
        self.collision_checker = collision_checker
        self.goal_checker = goal_checker
        self.sim_states_list = []
        self.disturbances_list = []
        self.crash = 0

        # Performance metrics
        self.timeouts = 0
        self.omegas = []  # weights in ESS, i.e. the w_i in \sum^N_{i=1} w_i \cdot \mu_i
        self.ctrl_hz = 0
        self.trajectories_list = []
        self.opt_trajs = []

        self.localize_jit = None
        self.vmap_localize_jit = None

    def initialize_from_config(self, config_data, section_name):
        Logger.initialize_from_config(self, config_data, section_name)
        if config_data.has_option(section_name, "goal_checker"):
            goal_checker_section_name = config_data.get(section_name, "goal_checker")
            self.goal_checker = factory_from_config(
                goal_checker_factory_base, config_data, goal_checker_section_name
            )
        if config_data.has_option(section_name, "collision_checker"):
            collision_checker_section_name = config_data.get(
                section_name, "collision_checker"
            )
            self.collision_checker = factory_from_config(
                collision_checker_factory_base,
                config_data,
                collision_checker_section_name,
            )
        self.batch_code = config_data.get(section_name, "batch_code")
        _, current_experiment_dir = self.create_save_dir()
        self.log_file_path = (
            current_experiment_dir + "/" + self.experiment_name + "_log"
        )

        if config_data.has_option(section_name, "num_states"):
            pass

    def create_save_dir(self):
        experiments_dir = self.experiment_root_dir + "/" + self.experiments_folder_name
        if not os.path.isdir(experiments_dir):
            os.mkdir(experiments_dir)
        current_experiment_dir = (
            self.experiment_root_dir
            + "/"
            + self.experiments_folder_name
            + "/"
            + self.batch_code
        )
        if not os.path.isdir(current_experiment_dir):
            os.mkdir(current_experiment_dir)
        return experiments_dir, current_experiment_dir

    def set_agent(self, agent):
        self.agent = agent

    def increment_timeouts(self):
        self.timeouts += 1

    def add_omega(self, omega):
        self.omegas.append(omega.squeeze())  # Expected shape of omega is (N,) for N num_trajectories

    def set_ctrl_hz(self, ctrl_hz):
        self.ctrl_hz = ctrl_hz

    def calculate_number_of_collisions(self, state, dynamics, collision_checker):
        if collision_checker.check(state):
            if not self.in_obstacle:
                self.number_of_collisions += 1
                self.in_obstacle = True
        else:
            self.in_obstacle = False

    def calculate_number_of_failures(self, state, dynamics, collision_checker):
        if collision_checker.check(state):
            self.number_of_failure = 1

    def calculate_agent_disturbance(self, state_next, old_state, action, nominal_dynamics):
        # For 3D drone, we track position and orientation disturbances
        pass

    def add_number_of_failure(self):
        self.number_of_failure += 1

    def append_state(self):
        state = self.agent.get_state()
        self.sim_states_list.append(state.reshape((-1, 1)))

    def log(self):
        timer = Timer.get_active()
        if self.number_of_failure > 0:
            self.crash = 1
        state = self.agent.get_state()
        self.sim_states_list.append(state.reshape((-1, 1)))
        if timer is not None:
            timer.stop().print_results()

    def add_trajectory_list(self, trajectory_list):
        self.trajectories_list.append(trajectory_list)

    def get_num_of_collisions(self):
        return copy.deepcopy(self.number_of_collisions)

    def get_num_of_failures(self):
        return copy.deepcopy(self.number_of_failure)

    @property
    def sim_states(self):
        return onp.concatenate(self.sim_states_list, axis=1)

    @property
    def n_sim_states(self):
        return len(self.sim_states_list)

    def shutdown(self, mean_eval_time=-1.0):
        # Calculate 3D-specific metrics
        if len(self.sim_states_list) > 0:
            # Position metrics: x, y, z
            mean_position = self.sim_states[:3, :].mean(axis=1)
            max_altitude = self.sim_states[2, :].max()  # max z
            min_altitude = self.sim_states[2, :].min()  # min z
            
            # Velocity metrics: vx, vy, vz
            velocity_magnitude = onp.sqrt(onp.sum(self.sim_states[3:6, :] ** 2, axis=0))
            mean_velocity = velocity_magnitude.mean()
            max_velocity = velocity_magnitude.max()
        else:
            mean_position = onp.array([0.0, 0.0, 0.0])
            max_altitude = 0.0
            min_altitude = 0.0
            mean_velocity = 0.0
            max_velocity = 0.0

        print(f"Logging to {self.log_file_path}")
        np.savez(
            self.log_file_path,
            states=self.sim_states,
            collisions=self.number_of_collisions,
            crash=self.crash,
            lap_time=self.agent.get_time(),
            mean_position=mean_position,
            max_altitude=max_altitude,
            min_altitude=min_altitude,
            mean_velocity=mean_velocity,
            max_velocity=max_velocity,
            mean_eval_time=mean_eval_time,
            timeouts=self.timeouts,
            omegas=self.omegas,
            ctrl_hz=self.ctrl_hz,
        )


class AutorallyCSSMPCLogger(AutorallyLogger):
    def __init__(self, experiment_dir=None, collision_checker=None, goal_checker=None):
        AutorallyLogger.__init__(self, experiment_dir)
        self.experiments_folder_name = "Autorally_CSSMPC_experiments"

    def initialize_from_config(self, config_data, section_name):
        AutorallyLogger.initialize_from_config(self, config_data, section_name)
