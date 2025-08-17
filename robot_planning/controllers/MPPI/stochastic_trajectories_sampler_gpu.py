import numpy as np
import ast
import copy
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import noise_sampler_factory_base
import multiprocessing as mp
from robot_planning.factory.factories import covariance_steering_helper_factory_base
from robot_planning.controllers.MPPI.stochastic_trajectories_sampler import (
    StochasticTrajectoriesSampler,
)
from robot_planning.factory.factories import dynamics_factory_base
from robot_planning.factory.factories import cost_evaluator_factory_base
from robot_planning.factory.factories import goal_checker_factory_base
import threading
import cvxpy

import os
import pycuda.autoinit

global drv
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from math import ceil
from time import time, sleep


class MPPIStochasticTrajectoriesSamplerGpu(StochasticTrajectoriesSampler):
    def __init__(
        self,
        number_of_trajectories=None,
        uncontrolled_trajectories_portion=None,
        noise_sampler=None,
    ):
        StochasticTrajectoriesSampler.__init__(
            self,
            number_of_trajectories,
            uncontrolled_trajectories_portion,
            noise_sampler,
        )
        self.print_info("in use")

    def initialize_from_config(self, config_data, section_name):
        StochasticTrajectoriesSampler.initialize_from_config(
            self, config_data, section_name
        )
        self.config_data = config_data
        self.section_name = section_name
        self.sample_count = self.number_of_trajectories
        params = ["control_dim", "state_dim", "horizon", "curand_kernel_n"]
        for param in params:
            setattr(
                self, param, int(ast.literal_eval(config_data.get(section_name, param)))
            )

        # initialize pycuda modeul
        self.initialize_pycuda()

    def initialize_pycuda(self):
        # load cuda source code
        folder = os.path.dirname(os.path.abspath(__file__))
        cuda_filename = folder + "/autorally_mppi.cu"
        with open(cuda_filename, "r") as f:
            code = f.read()
        cuda_file_parameters = {
            "CURAND_KERNEL_N": self.curand_kernel_n,
            "SAMPLE_COUNT": self.sample_count,
            "HORIZON": self.horizon,
            "CONTROL_DIM": self.control_dim,
        }

        # load dynamics parameters
        config_data = self.config_data
        dynamics_section_name = config_data.get(self.section_name, "dynamics")
        # can we avoid this unnecessary instantiation?
        self.dynamics = factory_from_config(
            dynamics_factory_base, config_data, dynamics_section_name
        )
        m_Vehicle_m = config_data.getfloat(dynamics_section_name, "m")
        m_Vehicle_Iz = config_data.getfloat(dynamics_section_name, "Iz")
        m_Vehicle_lF = config_data.getfloat(dynamics_section_name, "lF")
        m_Vehicle_lFR = config_data.getfloat(dynamics_section_name, "lFR")
        m_Vehicle_IwF = config_data.getfloat(dynamics_section_name, "IwF")
        m_Vehicle_IwR = config_data.getfloat(dynamics_section_name, "IwR")
        m_Vehicle_lR = m_Vehicle_lFR - m_Vehicle_lF
        m_Vehicle_rF = config_data.getfloat(dynamics_section_name, "rF")
        m_Vehicle_rR = config_data.getfloat(dynamics_section_name, "rR")
        m_Vehicle_h = config_data.getfloat(dynamics_section_name, "h")
        m_Vehicle_tire_B = config_data.getfloat(dynamics_section_name, "tire_B")
        m_Vehicle_tire_C = config_data.getfloat(dynamics_section_name, "tire_C")
        m_Vehicle_tire_D = config_data.getfloat(dynamics_section_name, "tire_D")
        m_Vehicle_kSteering = config_data.getfloat(dynamics_section_name, "kSteering")
        m_Vehicle_cSteering = config_data.getfloat(dynamics_section_name, "cSteering")
        # update to cuda file param
        cuda_file_parameters.update({"m_Vehicle_m": m_Vehicle_m})
        cuda_file_parameters.update({"m_Vehicle_Iz": m_Vehicle_Iz})
        cuda_file_parameters.update({"m_Vehicle_lF": m_Vehicle_lF})
        cuda_file_parameters.update({"m_Vehicle_lFR": m_Vehicle_lFR})
        cuda_file_parameters.update({"m_Vehicle_IwF": m_Vehicle_IwF})
        cuda_file_parameters.update({"m_Vehicle_IwR": m_Vehicle_IwR})
        cuda_file_parameters.update({"m_Vehicle_lR": m_Vehicle_lR})
        cuda_file_parameters.update({"m_Vehicle_rF": m_Vehicle_rF})
        cuda_file_parameters.update({"m_Vehicle_rR": m_Vehicle_rR})
        cuda_file_parameters.update({"m_Vehicle_h": m_Vehicle_h})
        cuda_file_parameters.update({"m_Vehicle_tire_B": m_Vehicle_tire_B})
        cuda_file_parameters.update({"m_Vehicle_tire_C": m_Vehicle_tire_C})
        cuda_file_parameters.update({"m_Vehicle_tire_D": m_Vehicle_tire_D})
        cuda_file_parameters.update({"m_Vehicle_kSteering": m_Vehicle_kSteering})
        cuda_file_parameters.update({"m_Vehicle_cSteering": m_Vehicle_cSteering})

        # load track map
        # size: 330,2
        self.track_points = np.moveaxis(self.dynamics.track.p, 0, 1).copy()
        """
        import matplotlib.pyplot as plt
        plt.plot(self.track_points[:,0],self.track_points[:,1],'*')
        plt.show()
        breakpoint()
        """

        self.device_track_points = self.to_device(self.track_points)
        # size: 330
        self.track_distance = np.hstack([[0], self.dynamics.track.s])
        self.device_track_distance = self.to_device(self.track_distance)

        # load cost matrix
        cost_evaluator_section_name = config_data.get(
            self.section_name, "cost_evaluator"
        )
        self.cost_evaluator = factory_from_config(
            cost_evaluator_factory_base, config_data, cost_evaluator_section_name
        )
        self.collision_checker = self.cost_evaluator.collision_checker
        self.track_width = self.collision_checker.track_width
        cuda_file_parameters.update({"TRACK_WIDTH": self.track_width})
        self.R = self.cost_evaluator.R
        self.Q = self.cost_evaluator.Q
        if hasattr(self.cost_evaluator, "QN"):
            self.QN = self.cost_evaluator.QN
        else:
            self.print_info("QN not specified in config, using QN:=Q")
            self.QN = self.cost_evaluator.Q
        self.device_R = self.to_device(self.R)
        self.device_Q = self.to_device(self.Q)
        self.device_QN = self.to_device(self.QN)
        cuda_file_parameters.update({"TRACK_INFO_SIZE": self.track_points.shape[0]})

        # load goal state
        goal_checker_section_name = config_data.get(self.section_name, "goal_checker")
        self.goal_checker = factory_from_config(
            goal_checker_factory_base, config_data, goal_checker_section_name
        )
        self.goal_state = self.goal_checker.goal_state
        self.device_goal_state = self.to_device(self.goal_state)

        # compile cuda code with file parameters
        mod = SourceModule(code % cuda_file_parameters, no_extern_c=True)
        if self.sample_count < 1024:
            # if sample size is small only employ one grid
            self.cuda_block_size = (self.sample_count, 1, 1)
            self.cuda_grid_size = (1, 1)
        else:
            # employ multiple grid,
            self.cuda_block_size = (1024, 1, 1)
            self.cuda_grid_size = (ceil(self.sample_count / 1024.0), 1)
        self.print_info(
            "cuda block size %d, grid size %d"
            % (self.cuda_block_size[0], self.cuda_grid_size[0])
        )

        # register cuda functions
        self.cuda_init_curand_kernel = mod.get_function("init_curand_kernel")
        self.cuda_generate_gaussian_noise = mod.get_function("generate_gaussian_noise")
        self.cuda_propagate_dynamics = mod.get_function("propagate_dynamics")
        self.cuda_set_cost_matrices = mod.get_function("set_cost_matrices")
        self.cuda_set_track_info = mod.get_function("set_track_info")
        self.cuda_set_goal_state = mod.get_function("set_goal_state")
        self.cuda_evaluate_trajectory_cost = mod.get_function(
            "evaluate_trajectory_cost"
        )

        assert (
            int(self.cuda_init_curand_kernel.num_regs * self.cuda_block_size[0])
            <= 65536
        )
        assert (
            int(self.cuda_generate_gaussian_noise.num_regs * self.cuda_block_size[0])
            <= 65536
        )
        assert (
            int(self.cuda_propagate_dynamics.num_regs * self.cuda_block_size[0])
            <= 65536
        )
        assert int(self.cuda_set_cost_matrices.num_regs) <= 64
        assert int(self.cuda_set_track_info.num_regs) <= 64
        assert int(self.cuda_set_goal_state.num_regs) <= 64
        # XXX this is breached if printf is removed from the code, weird
        assert (
            int(self.cuda_evaluate_trajectory_cost.num_regs * self.cuda_block_size[0])
            <= 65536
        )
        self.print_info(
            "cuda_init_curand_kernel.num_regs = %d"
            % (self.cuda_init_curand_kernel.num_regs)
        )
        self.print_info(
            "cuda_generate_gaussian_noise.num_regs = %d"
            % (self.cuda_generate_gaussian_noise.num_regs)
        )
        self.print_info(
            "cuda_propagate_dynamics.num_regs = %d"
            % (self.cuda_propagate_dynamics.num_regs)
        )
        self.print_info(
            "cuda_evaluate_trajectory_cost.num_regs = %d"
            % (self.cuda_evaluate_trajectory_cost.num_regs)
        )

        # initialize GPU data structure
        self.cuda_set_cost_matrices(
            self.device_Q,
            self.device_R,
            self.device_QN,
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )
        self.cuda_set_track_info(
            self.device_track_points,
            self.device_track_distance,
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )
        self.cuda_set_goal_state(
            self.device_goal_state, block=(1, 1, 1), grid=(1, 1, 1)
        )

        # initialize curand (random number generator)
        seed = np.int32(int(time() * 10000))
        self.cuda_init_curand_kernel(
            seed, block=(self.curand_kernel_n, 1, 1), grid=(1, 1, 1)
        )

        self.rand_vals = np.zeros(
            (self.sample_count, (self.horizon - 1), self.control_dim), dtype=np.float32
        )
        self.device_rand_vals = self.to_device(self.rand_vals)

        self.trajectory = np.zeros(
            (self.sample_count, self.horizon, self.state_dim), dtype=np.float32
        )
        self.device_trajectory = self.to_device(self.trajectory)

        self.device_mean = self.to_device(self.noise_sampler.mean)
        self.device_covariance = self.to_device(self.noise_sampler.covariance)

        sleep(1)

    def sampleCPU(
        self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator
    ):
        #  state_cur is the current state, v is the nominal control sequence
        state_start = state_cur.copy()
        costs = np.zeros((self.number_of_trajectories, 1, 1))
        state_cur = np.tile(
            state_start.reshape((-1, 1)), (1, self.number_of_trajectories)
        )
        trajectories = np.zeros(
            (self.number_of_trajectories, state_cur.shape[0], control_horizon)
        )
        trajectories[:, :, 0] = np.swapaxes(state_cur, 0, 1)
        noises = self.noise_sampler.sample(
            control_dim, (control_horizon - 1) * self.number_of_trajectories
        )
        noises = noises.reshape(
            (control_dim, (control_horizon - 1), self.number_of_trajectories)
        )
        num_controlled_trajectories = int(
            (1 - self.uncontrolled_trajectories_portion) * self.number_of_trajectories
        )
        us = np.zeros((v.shape[0], v.shape[1], self.number_of_trajectories))
        us[:, :, :num_controlled_trajectories] = np.expand_dims(v, axis=2)
        us += noises
        for j in range(control_horizon - 1):
            costs += cost_evaluator.evaluate(
                state_cur, us[:, j, :], noises[:, j, :], dynamics=dynamics
            )
            state_cur = dynamics.propagate(state_cur, us[:, j, :])
            trajectories[:, :, j + 1] = np.swapaxes(state_cur, 0, 1)
        costs += cost_evaluator.evaluate_terminal_cost(state_cur, dynamics=dynamics)
        us = np.moveaxis(us, 2, 0)
        return trajectories, us, costs

    def sample(
        self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator
    ):
        t = time()
        assert control_horizon == self.horizon
        assert control_dim == self.control_dim
        #  state_cur is the current state, v is the nominal control sequence
        # generate noise, propagate dynamics, and evaluate cost
        self.cuda_generate_gaussian_noise(
            self.device_mean,
            self.device_covariance,
            self.device_rand_vals,
            block=(self.curand_kernel_n, 1, 1),
            grid=(1, 1, 1),
        )
        # target_noise_dim = (control_dim, (control_horizon - 1), self.number_of_trajectories)
        noise_dim = (self.number_of_trajectories, (control_horizon - 1), control_dim)
        noises = self.from_device(self.device_rand_vals, noise_dim)
        num_controlled_trajectories = int(
            (1 - self.uncontrolled_trajectories_portion) * self.number_of_trajectories
        )
        x0 = state_cur.copy()

        # CPU implementation
        """
        t = time()
        cpu_noises = np.moveaxis(noises, 2, 0)
        cpu_noises = np.moveaxis(cpu_noises, 2, 1)
        cpu_state_start = state_cur.copy()
        cpu_costs = np.zeros((self.number_of_trajectories, 1, 1))
        state_cur = np.tile(cpu_state_start.reshape((-1, 1)), (1, self.number_of_trajectories))
        cpu_trajectories = np.zeros((self.number_of_trajectories, state_cur.shape[0], control_horizon))
        cpu_trajectories[:, :, 0] = np.swapaxes(state_cur, 0, 1)

        cpu_us = np.zeros((v.shape[0], v.shape[1], self.number_of_trajectories))
        cpu_us[:, :, :num_controlled_trajectories] = np.expand_dims(v, axis=2)
        cpu_us += cpu_noises
        for j in range(control_horizon-1):
            cpu_costs += cost_evaluator.evaluate(state_cur, cpu_us[:, j, :], cpu_noises[:, j, :], dynamics=dynamics)
            state_cur = dynamics.propagate_vv(state_cur, cpu_us[:, j, :])
            cpu_trajectories[:, :, j+1] = np.swapaxes(state_cur, 0, 1)
        cpu_costs += cost_evaluator.evaluate_terminal_cost(state_cur, dynamics=dynamics)
        dt = time()-t
        #print("cpu time: %.5f, %.2fHz"%(dt,1/dt))
        """

        # GPU implementation
        us = np.moveaxis(noises, 2, 0)
        us = np.moveaxis(us, 2, 1).copy()
        us[:, :, :num_controlled_trajectories] += np.expand_dims(v, axis=2)

        state_start = x0.copy()
        v = np.moveaxis(v, 0, 1)  # dim:19:2
        trajectories = np.zeros(
            self.number_of_trajectories * state_cur.shape[0] * control_horizon,
            dtype=np.float32,
        )
        device_state_start = self.to_device(state_start)
        device_v = self.to_device(v)

        self.cuda_propagate_dynamics(
            device_state_start,
            device_v,
            self.device_rand_vals,
            drv.Out(trajectories),
            np.int32(num_controlled_trajectories),
            block=self.cuda_block_size,
            grid=self.cuda_grid_size,
        )
        trajectories = trajectories.reshape(
            self.number_of_trajectories, self.horizon, self.state_dim
        )
        device_trajectory = self.to_device(trajectories)
        costs = np.zeros(self.number_of_trajectories, dtype=np.float32)
        self.cuda_evaluate_trajectory_cost(
            device_trajectory,
            device_v,
            self.device_rand_vals,
            drv.Out(costs),
            np.int32(num_controlled_trajectories),
            block=self.cuda_block_size,
            grid=self.cuda_grid_size,
        )
        trajectories = np.swapaxes(trajectories, 1, 2)

        # calculate cost for gpu trajectories on cpu
        """
        cpu_costs = np.zeros((self.number_of_trajectories, 1, 1))
        for j in range(control_horizon-1):
            step_cost = cost_evaluator.evaluate(trajectories[:,:,j].T, us[:, j, :], noises[:, j, :].T, dynamics=dynamics)
            cpu_costs += step_cost

        cpu_terminal_cost = cost_evaluator.evaluate_terminal_cost(trajectories[:,:,-1].T, dynamics=dynamics)
        cpu_costs += cpu_terminal_cost
        print("cost diff = %.2f%%"%(np.linalg.norm(cpu_costs.flatten()-costs.flatten())/np.linalg.norm(costs.flatten())*100))
        state = trajectories[50,:,-1].T
        cpu_terminal_cost = cost_evaluator.evaluate_terminal_cost(trajectories[:,:,-1].T, dynamics=dynamics)
        e_psi, e_y, s, index = dynamics.track.localize_one(np.array((state[-2], state[-1])), state[-3],return_index=True)
        print("cpu: %d, %.3f, %.3f, %.3f, cost=%.2f"%(index,e_psi,e_y,s,cpu_terminal_cost[50]))
        """

        costs = costs.reshape(costs.shape[-1], 1, 1)
        """
        cpu_best = np.argmin(cpu_costs.flatten())
        gpu_best = np.argmin(costs.flatten())
        if (cpu_best != gpu_best):
            print("unmatched best...")
        """
        us = np.moveaxis(us, 2, 0)

        # costs = cpu_costs
        # trajectories = cpu_trajectories
        dt = time() - t
        self.print_info("control freq MPPI: %.2fHz" % (1.0 / dt))
        return trajectories, us, costs

    def to_device(self, data):
        return drv.to_device(np.array(data, dtype=np.float32).flatten())

    def from_device(self, data, shape, dtype=np.float32):
        return drv.from_device(data, shape, dtype)


class CCMPPIStochasticTrajectoriesSamplerGpu(StochasticTrajectoriesSampler):
    def __init__(
        self,
        number_of_trajectories=None,
        uncontrolled_trajectories_portion=None,
        noise_sampler=None,
    ):
        StochasticTrajectoriesSampler.__init__(
            self,
            number_of_trajectories,
            uncontrolled_trajectories_portion,
            noise_sampler,
        )
        self.print_info("in use")

    def initialize_from_config(self, config_data, section_name):
        StochasticTrajectoriesSampler.initialize_from_config(
            self, config_data, section_name
        )
        # CC specic
        covariance_steering_helper_section_name = config_data.get(
            section_name, "covariance_steering_helper"
        )
        self.covariance_steering_helper = factory_from_config(
            covariance_steering_helper_factory_base,
            config_data,
            covariance_steering_helper_section_name,
        )
        self.config_data = config_data
        self.section_name = section_name
        self.sample_count = self.number_of_trajectories
        params = ["control_dim", "state_dim", "horizon", "curand_kernel_n"]
        for param in params:
            setattr(
                self, param, int(ast.literal_eval(config_data.get(section_name, param)))
            )
        if self.config_data.has_option(section_name, "static_k"):
            setattr(
                self,
                "static_k",
                ast.literal_eval(config_data.get(section_name, "static_k")),
            )
            self.print_info("use static K for CC:", self.static_k)
        else:
            self.static_k = False

        # initialize pycuda modeul
        self.initialize_pycuda()
        if self.static_k:
            self.prepare_static_ABK()

    def initialize_pycuda(self):
        # load cuda source code
        folder = os.path.dirname(os.path.abspath(__file__))
        cuda_filename = folder + "/autorally_ccmppi.cu"
        with open(cuda_filename, "r") as f:
            code = f.read()
        cuda_file_parameters = {
            "CURAND_KERNEL_N": self.curand_kernel_n,
            "SAMPLE_COUNT": self.sample_count,
            "HORIZON": self.horizon,
            "CONTROL_DIM": self.control_dim,
        }

        # load dynamics parameters
        config_data = self.config_data
        dynamics_section_name = config_data.get(self.section_name, "dynamics")
        # can we avoid this unnecessary instantiation?
        self.dynamics = factory_from_config(
            dynamics_factory_base, config_data, dynamics_section_name
        )
        m_Vehicle_m = config_data.getfloat(dynamics_section_name, "m")
        m_Vehicle_Iz = config_data.getfloat(dynamics_section_name, "Iz")
        m_Vehicle_lF = config_data.getfloat(dynamics_section_name, "lF")
        m_Vehicle_lFR = config_data.getfloat(dynamics_section_name, "lFR")
        m_Vehicle_IwF = config_data.getfloat(dynamics_section_name, "IwF")
        m_Vehicle_IwR = config_data.getfloat(dynamics_section_name, "IwR")
        m_Vehicle_lR = m_Vehicle_lFR - m_Vehicle_lF
        m_Vehicle_rF = config_data.getfloat(dynamics_section_name, "rF")
        m_Vehicle_rR = config_data.getfloat(dynamics_section_name, "rR")
        m_Vehicle_h = config_data.getfloat(dynamics_section_name, "h")
        m_Vehicle_tire_B = config_data.getfloat(dynamics_section_name, "tire_B")
        m_Vehicle_tire_C = config_data.getfloat(dynamics_section_name, "tire_C")
        m_Vehicle_tire_D = config_data.getfloat(dynamics_section_name, "tire_D")
        m_Vehicle_kSteering = config_data.getfloat(dynamics_section_name, "kSteering")
        m_Vehicle_cSteering = config_data.getfloat(dynamics_section_name, "cSteering")
        # update to cuda file param
        cuda_file_parameters.update({"m_Vehicle_m": m_Vehicle_m})
        cuda_file_parameters.update({"m_Vehicle_Iz": m_Vehicle_Iz})
        cuda_file_parameters.update({"m_Vehicle_lF": m_Vehicle_lF})
        cuda_file_parameters.update({"m_Vehicle_lFR": m_Vehicle_lFR})
        cuda_file_parameters.update({"m_Vehicle_IwF": m_Vehicle_IwF})
        cuda_file_parameters.update({"m_Vehicle_IwR": m_Vehicle_IwR})
        cuda_file_parameters.update({"m_Vehicle_lR": m_Vehicle_lR})
        cuda_file_parameters.update({"m_Vehicle_rF": m_Vehicle_rF})
        cuda_file_parameters.update({"m_Vehicle_rR": m_Vehicle_rR})
        cuda_file_parameters.update({"m_Vehicle_h": m_Vehicle_h})
        cuda_file_parameters.update({"m_Vehicle_tire_B": m_Vehicle_tire_B})
        cuda_file_parameters.update({"m_Vehicle_tire_C": m_Vehicle_tire_C})
        cuda_file_parameters.update({"m_Vehicle_tire_D": m_Vehicle_tire_D})
        cuda_file_parameters.update({"m_Vehicle_kSteering": m_Vehicle_kSteering})
        cuda_file_parameters.update({"m_Vehicle_cSteering": m_Vehicle_cSteering})

        # load track map
        # size: 330,2
        self.track_points = np.moveaxis(self.dynamics.track.p, 0, 1).copy()
        """
        import matplotlib.pyplot as plt
        plt.plot(self.track_points[:,0],self.track_points[:,1],'*')
        plt.show()
        breakpoint()
        """

        self.device_track_points = self.to_device(self.track_points)
        # size: 330
        self.track_distance = np.hstack([[0], self.dynamics.track.s])
        self.device_track_distance = self.to_device(self.track_distance)

        # load cost matrix
        cost_evaluator_section_name = config_data.get(
            self.section_name, "cost_evaluator"
        )
        self.cost_evaluator = factory_from_config(
            cost_evaluator_factory_base, config_data, cost_evaluator_section_name
        )
        self.collision_checker = self.cost_evaluator.collision_checker
        self.track_width = self.collision_checker.track_width
        cuda_file_parameters.update({"TRACK_WIDTH": self.track_width})
        self.R = self.cost_evaluator.R
        self.Q = self.cost_evaluator.Q
        if hasattr(self.cost_evaluator, "QN"):
            self.QN = self.cost_evaluator.QN
        else:
            self.print_info("QN not specified in config, using QN:=Q")
            self.QN = self.cost_evaluator.Q
        self.device_R = self.to_device(self.R)
        self.device_Q = self.to_device(self.Q)
        self.device_QN = self.to_device(self.QN)
        cuda_file_parameters.update({"TRACK_INFO_SIZE": self.track_points.shape[0]})

        # load goal state
        goal_checker_section_name = config_data.get(self.section_name, "goal_checker")
        self.goal_checker = factory_from_config(
            goal_checker_factory_base, config_data, goal_checker_section_name
        )
        self.goal_state = self.goal_checker.goal_state
        self.device_goal_state = self.to_device(self.goal_state)

        # compile cuda code with file parameters
        mod = SourceModule(code % cuda_file_parameters, no_extern_c=True)
        if self.sample_count < 1024:
            # if sample size is small only employ one grid
            self.cuda_block_size = (self.sample_count, 1, 1)
            self.cuda_grid_size = (1, 1)
        else:
            # employ multiple grid,
            self.cuda_block_size = (1024, 1, 1)
            self.cuda_grid_size = (ceil(self.sample_count / 1024.0), 1)
        self.print_info(
            "cuda block size %d, grid size %d"
            % (self.cuda_block_size[0], self.cuda_grid_size[0])
        )

        # register cuda functions
        self.cuda_init_curand_kernel = mod.get_function("init_curand_kernel")
        self.cuda_generate_gaussian_noise = mod.get_function("generate_gaussian_noise")
        self.cuda_propagate_dynamics = mod.get_function("propagate_dynamics")
        self.cuda_set_cost_matrices = mod.get_function("set_cost_matrices")
        self.cuda_set_track_info = mod.get_function("set_track_info")
        self.cuda_set_goal_state = mod.get_function("set_goal_state")
        self.cuda_evaluate_trajectory_cost = mod.get_function(
            "evaluate_trajectory_cost"
        )

        assert (
            int(self.cuda_init_curand_kernel.num_regs * self.cuda_block_size[0])
            <= 65536
        )
        assert (
            int(self.cuda_generate_gaussian_noise.num_regs * self.cuda_block_size[0])
            <= 65536
        )
        # assert int(self.cuda_propagate_dynamics.num_regs * self.cuda_block_size[0]) <= 65536
        assert int(self.cuda_set_cost_matrices.num_regs) <= 64
        assert int(self.cuda_set_track_info.num_regs) <= 64
        assert int(self.cuda_set_goal_state.num_regs) <= 64
        # XXX this is breached if printf is removed from the code, weird
        # assert int(self.cuda_evaluate_trajectory_cost.num_regs * self.cuda_block_size[0]) <= 65536
        self.print_info(
            "cuda_init_curand_kernel.num_regs = %d"
            % (self.cuda_init_curand_kernel.num_regs)
        )
        self.print_info(
            "cuda_generate_gaussian_noise.num_regs = %d"
            % (self.cuda_generate_gaussian_noise.num_regs)
        )
        self.print_info(
            "cuda_propagate_dynamics.num_regs = %d"
            % (self.cuda_propagate_dynamics.num_regs)
        )
        self.print_info(
            "cuda_evaluate_trajectory_cost.num_regs = %d"
            % (self.cuda_evaluate_trajectory_cost.num_regs)
        )

        # initialize GPU data structure
        self.cuda_set_cost_matrices(
            self.device_Q,
            self.device_R,
            self.device_QN,
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )
        self.cuda_set_track_info(
            self.device_track_points,
            self.device_track_distance,
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )
        self.cuda_set_goal_state(
            self.device_goal_state, block=(1, 1, 1), grid=(1, 1, 1)
        )

        # initialize curand (random number generator)
        seed = np.int32(int(time() * 10000))
        self.cuda_init_curand_kernel(
            seed, block=(self.curand_kernel_n, 1, 1), grid=(1, 1, 1)
        )

        self.rand_vals = np.zeros(
            (self.sample_count, (self.horizon - 1), self.control_dim), dtype=np.float32
        )
        self.device_rand_vals = self.to_device(self.rand_vals)

        self.trajectory = np.zeros(
            (self.sample_count, self.horizon, self.state_dim), dtype=np.float32
        )
        self.device_trajectory = self.to_device(self.trajectory)

        self.device_mean = self.to_device(self.noise_sampler.mean)
        self.device_covariance = self.to_device(self.noise_sampler.covariance)

        sleep(1)

    def sampleCPU(
        self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator
    ):
        #  state_cur is the current state, v is the nominal control sequence
        state_start = state_cur.copy()
        costs = np.zeros((self.number_of_trajectories, 1, 1))
        state_cur = np.tile(
            state_start.reshape((-1, 1)), (1, self.number_of_trajectories)
        )
        trajectories = np.zeros(
            (self.number_of_trajectories, state_cur.shape[0], control_horizon)
        )
        trajectories[:, :, 0] = np.swapaxes(state_cur, 0, 1)
        noises = self.noise_sampler.sample(
            control_dim, (control_horizon - 1) * self.number_of_trajectories
        )
        noises = noises.reshape(
            (control_dim, (control_horizon - 1), self.number_of_trajectories)
        )
        num_controlled_trajectories = int(
            (1 - self.uncontrolled_trajectories_portion) * self.number_of_trajectories
        )
        us = np.zeros((v.shape[0], v.shape[1], self.number_of_trajectories))
        us[:, :, :num_controlled_trajectories] = np.expand_dims(v, axis=2)
        us += noises
        for j in range(control_horizon - 1):
            costs += cost_evaluator.evaluate(
                state_cur, us[:, j, :], noises[:, j, :], dynamics=dynamics
            )
            state_cur = dynamics.propagate(state_cur, us[:, j, :])
            trajectories[:, :, j + 1] = np.swapaxes(state_cur, 0, 1)
        costs += cost_evaluator.evaluate_terminal_cost(state_cur, dynamics=dynamics)
        us = np.moveaxis(us, 2, 0)
        return trajectories, us, costs

    def prepare_static_ABK(self):
        dynamics = self.dynamics
        self.covariance_steering_helper.dynamics_linearizer.set_dynamics(dynamics)
        # vx,vy,wz,wF,wR, psi,X,Y(cartesian)
        # x0 = np.array([6,0,0,20,20,0,0,0],dtype=np.float)
        x0 = np.array([6, 0, 0, 0, 0, 0, 0, 0], dtype=np.float)
        v = np.zeros((self.control_dim, self.horizon - 1))
        reference_trajectory = self.rollout_out(x0, v, dynamics)
        try:
            Ks, As, Bs, _, _, _ = self.covariance_steering_helper.covariance_control(
                state=x0.T,
                ref_state_vec=reference_trajectory.T,
                ref_ctrl_vec=v.T,
                return_sx=True,
                Sigma_epsilon=self.noise_sampler.covariance,
            )
            self.print_info("static ABK created")
        except cvxpy.error.SolverError:
            self.print_warning("Can't find solution to CC, fallback to mppi")
            Ks = np.zeros((self.horizon - 1, self.control_dim, self.state_dim))
            As = np.zeros((self.horizon - 1, self.state_dim, self.state_dim))
            Bs = np.zeros((self.horizon - 1, self.state_dim, self.control_dim))
        self.Ks = Ks
        self.As = As
        self.Bs = Bs
        return

    # CCMPPI
    def sample(
        self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator
    ):
        t = time()
        assert control_horizon == self.horizon
        assert control_dim == self.control_dim
        #  state_cur is the current state, v is the nominal control sequence
        # generate noise, propagate dynamics, and evaluate cost
        self.cuda_generate_gaussian_noise(
            self.device_mean,
            self.device_covariance,
            self.device_rand_vals,
            block=(self.curand_kernel_n, 1, 1),
            grid=(1, 1, 1),
        )
        # target_noise_dim = (control_dim, (control_horizon - 1), self.number_of_trajectories)
        noise_dim = (self.number_of_trajectories, (control_horizon - 1), control_dim)
        noises = self.from_device(self.device_rand_vals, noise_dim)
        num_controlled_trajectories = int(
            (1 - self.uncontrolled_trajectories_portion) * self.number_of_trajectories
        )
        x0 = state_cur.copy()

        if self.static_k:
            Ks = self.Ks
            As = self.As
            Bs = self.Bs
        else:
            self.covariance_steering_helper.dynamics_linearizer.set_dynamics(dynamics)
            state_start = state_cur.copy()
            reference_trajectory = self.rollout_out(state_start, v, dynamics)
            try:
                (
                    Ks,
                    As,
                    Bs,
                    _,
                    _,
                    _,
                ) = self.covariance_steering_helper.covariance_control(
                    state=state_cur.T,
                    ref_state_vec=reference_trajectory.T,
                    ref_ctrl_vec=v.T,
                    return_sx=True,
                    Sigma_epsilon=self.noise_sampler.covariance,
                )
            except cvxpy.error.SolverError:
                self.print_warning("Can't find solution to CC, fallback to mppi")
                Ks = np.zeros((self.horizon - 1, self.control_dim, self.state_dim))
                As = np.zeros((self.horizon - 1, self.state_dim, self.state_dim))
                Bs = np.zeros((self.horizon - 1, self.state_dim, self.control_dim))
        y = np.zeros((num_controlled_trajectories, self.state_dim, 1))
        device_Ks = self.to_device(Ks)
        device_As = self.to_device(As)
        device_Bs = self.to_device(Bs)

        # GPU implementation
        us = np.moveaxis(noises, 2, 0)
        us = np.moveaxis(us, 2, 1).copy()
        us[:, :, :num_controlled_trajectories] += np.expand_dims(v, axis=2)

        state_start = x0.copy()
        v = np.moveaxis(v, 0, 1)  # dim:19:2
        trajectories = np.zeros(
            self.number_of_trajectories * state_cur.shape[0] * control_horizon,
            dtype=np.float32,
        )
        device_state_start = self.to_device(state_start)
        device_v = self.to_device(v)

        self.cuda_propagate_dynamics(
            device_state_start,
            device_v,
            self.device_rand_vals,
            drv.Out(trajectories),
            np.int32(num_controlled_trajectories),
            device_Ks,
            device_As,
            device_Bs,
            block=self.cuda_block_size,
            grid=self.cuda_grid_size,
        )
        trajectories = trajectories.reshape(
            self.number_of_trajectories, self.horizon, self.state_dim
        )
        device_trajectory = self.to_device(trajectories)
        costs = np.zeros(self.number_of_trajectories, dtype=np.float32)
        self.cuda_evaluate_trajectory_cost(
            device_trajectory,
            device_v,
            self.device_rand_vals,
            drv.Out(costs),
            np.int32(num_controlled_trajectories),
            block=self.cuda_block_size,
            grid=self.cuda_grid_size,
        )
        trajectories = np.swapaxes(trajectories, 1, 2)

        costs = costs.reshape(costs.shape[-1], 1, 1)
        us = np.moveaxis(us, 2, 0)

        dt = time() - t
        self.print_info("control freq CCMPPI: %.2fHz" % (1.0 / dt))

        return trajectories, us, costs

    def rollout_out(self, state_cur, v, dynamics):
        trajectory = np.zeros((dynamics.get_state_dim()[0], v.shape[1] + 1))
        trajectory[:, 0] = state_cur
        for i in range(v.shape[1]):
            state_next = dynamics.propagate(state_cur, v[:, i])
            trajectory[:, i + 1] = state_next
            state_cur = state_next
        return trajectory

    def to_device(self, data):
        return drv.to_device(np.array(data, dtype=np.float32).flatten())

    def from_device(self, data, shape, dtype=np.float32):
        return drv.from_device(data, shape, dtype)
