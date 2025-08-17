"""
Run ablations for

Ours
Ours - neural CBF (=hand-tuned CBF) (includes efficient, hand-tuned CBF cost, and local repair)
Ours - Efficient --> (has neural CBF in cost)
Ours - Local repair

MPPI
MPPI + Neural CBF (in cost)
MPPI + Efficient (handtuned CBF)
MPPI + Local repair
"""
import shutil
from os import mkdir

import jax.numpy as np
import jax
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys

aa = np.arange(5)
import ipdb
import loguru
import numpy as onp
from robot_planning.batch_experimentation.loggers import AutorallyNpzLogger
from robot_planning.controllers.MPPI.MPPI import MPPI
from robot_planning.environment.cost_evaluators import AutorallyMPPICostEvaluator
from robot_planning.environment.goal_checker import AutorallyCartesianGoalChecker
from robot_planning.environment.robots.simulated_robot import SimulatedRobot
from robot_planning.helper.timer import Timer

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
import os
from itertools import product
from collections import deque
from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factories import renderer_factory_base
from robot_planning.factory.factories import logger_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import goal_checker_factory_base
from robot_planning.factory.factories import collision_checker_factory_base
from robot_planning.factory.factories import dynamics_factory_base
import unittest

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
jax.config.update('jax_platform_name', 'gpu')
# jax.config.update('jax_platform_name', 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

# # Print cuda visible devices
# print(os.environ["CUDA_VISIBLE_DEVICES"])

# print(jax.__path__)
# import os
# print(os.environ["LD_LIBRARY_PATH"])
# exit(0)

class TestRunAutorallyMPPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")

    def test_run_autorally_MPPI(self):
        # Make ablations directory if it doesn't exist
        if not os.path.exists('../experiments/Autorally_experiments/ablations'):
            mkdir('../experiments/Autorally_experiments/ablations')

        # for ablation in ["mppi", "mppi_plus_local_repair", "mppi_plus_neural_cbf", "mppi_plus_efficient", "ours_minus_local_repair"]:
        for ablation in ["smppi", "ours"]:
        # for ablation in ["mppi", "smppi", "smppi_plus_efficient"]:
        # for ablation in ["ours", "ours_minus_local_repair"]:
        # for ablation in ["ours", "ours_minus_neural_cbf", "ours_minus_efficient", "ours_minus_local_repair", "mppi", "mppi_plus_neural_cbf", "mppi_plus_efficient", "mppi_plus_local_repair"]:
            # Parameter set up
            experiment_ids = range(10)
            n_trajectories_setting = [30]
            control_horizon_setting = [5, 8, 11, 14, 17, 20, 23]
            # control_horizon_setting = [5, 8, 11, 14, 17, 20, 23]
            cbf_alpha_settings = [0.9]  # CBF alpha (controls aggression)  # Eric: might be of interest
            Q_vx = 50.0
            Q_epsi_setting = [0.0]  # Eric: error yaw (crucial)
            Q_ey_setting = [0.0]  # also try values 30.0 to 50.0  # Eric:track error lateral deviation (crucial)
            tgt_vels = [12.0]  # Eric: The velocity to track (target velocity) (below 8 and most controllers will do well, and higher is worse)

            render = True
            render_png = False
            log = False
            skip_previous_exps = True

            test_agents = ["ncbf"]

            np.set_printoptions(
                edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%10.3f" % x)
            )


            # config_path = "configs/run_Autorally_CBF_MPPI_for_experiment.cfg"
            # config_path = "configs/reverse_run_Autorally_CBF_MPPI_for_experiment.cfg"
            config_path = "configs/reverse_run_Autorally_ablation_with_repair.cfg"
            # config_path = "configs/reverse_run_Autorally_cluttered_env.cfg"

            # Eric: add ablation logic
            match ablation:
                case "ours":
                    pass
                case "ours_minus_neural_cbf":
                    test_agents = ["cbf"]
                case "ours_minus_efficient":
                    test_agents = ["ncbf_inefficient"]
                case "ours_minus_local_repair":
                    config_path = "configs/reverse_run_Autorally_ablation_no_repair.cfg"
                case "ours_minus_local_repair_no_soft":
                    test_agents = ["ncbf_no_soft"]
                    config_path = "configs/reverse_run_Autorally_ablation_no_repair.cfg"
                case "smppi":
                    test_agents = ["cbf_inefficient"]
                    # config_path = "configs/reverse_run_Autorally_ablation_no_repair.cfg"
                case "smppi_plus_efficient":
                    test_agents = ["cbf"]
                    # config_path = "configs/reverse_run_Autorally_ablation_no_repair.cfg"
                case "mppi":
                    test_agents = ["baseline"]
                    config_path = "configs/reverse_run_Autorally_ablation_no_repair.cfg"
                case "mppi_plus_neural_cbf":
                    test_agents = ["ncbf_inefficient"]
                    config_path = "configs/reverse_run_Autorally_ablation_no_repair.cfg"
                case "mppi_plus_efficient":
                    test_agents = ["cbf_no_soft"]
                    config_path = "configs/reverse_run_Autorally_ablation_no_repair.cfg"
                case "mppi_plus_local_repair":
                    test_agents = ["baseline"]
                case "CEM":
                    test_agents = ["CEM"]
                    config_path = "configs/reverse_run_Autorally_ablation_no_repair.cfg"
                case _:
                    raise NotImplementedError(f"Ablation not implemented: {ablation}")

            # config_path = "configs/reverse_run_Autorally_cluttered_env.cfg"

            config_data = ConfigParser.ConfigParser()
            config_data.read(config_path)

            experiments = product(
                test_agents,
                experiment_ids,
                n_trajectories_setting,
                control_horizon_setting,
                cbf_alpha_settings,
                Q_epsi_setting,
                Q_ey_setting,
                tgt_vels,
            )
            length = len(test_agents) * len(experiment_ids) * len(n_trajectories_setting) * len(control_horizon_setting) * len(cbf_alpha_settings) * len(Q_epsi_setting) * len(Q_ey_setting) * len(tgt_vels)
            for test_agent, experiment_index, n_traj, horizon, alpha, Q_epsi, Q_ey, tgt_vel in tqdm(experiments, desc='Running experiments...', position=0, leave=True, total=length, file=sys.stdout):
                print(f"{test_agent}: {(experiment_index, n_traj, horizon, alpha, Q_epsi, Q_ey, tgt_vel)}")
                if skip_previous_exps:
                    if os.path.exists(
                            f"../experiments/Autorally_experiments/ablations/{ablation}/{experiment_index}/{test_agent}_n_traj_{n_traj}_horizon_{horizon}_alpha_{alpha}_Qepsi_{Q_epsi}_Qey_{Q_ey}_tgtvel_{tgt_vel}_log.npz"):
                        print('Already ran this experiment. Skipping...')
                        continue

                loguru.logger.info(f'Running for file {experiment_index}/{test_agent}_n_traj_{n_traj}_horizon_{horizon}_alpha_{alpha}_Qepsi_{Q_epsi}_Qey_{Q_ey}_tgtvel_{tgt_vel}_log.npz')

                # Use the experiment_index as the seed.
                seed = 12345 + experiment_index
                traj_sampler_name = f"{test_agent}_stochastic_trajectories_sampler"
                noise_sampler_name = config_data.get(traj_sampler_name, "noise_sampler")
                config_data.set(noise_sampler_name, "seed", str(seed))

                # Update the batch code and experiment parameters
                # config_data.set("logger", "batch_code", str(experiment_index))
                os.makedirs(f"../experiments/Autorally_experiments/ablations/{ablation}", exist_ok=True)
                config_data.set("logger", "batch_code", f"ablations/{ablation}/{experiment_index}")
                config_data.set("logger", "ablation", f"{ablation}")
                config_data.set(
                    "logger",
                    "experiment_name",
                    f"{test_agent}_n_traj_{n_traj}_horizon_{horizon}_alpha_{alpha}_Qepsi_{Q_epsi}_Qey_{Q_ey}_tgtvel_{tgt_vel}",
                )

                if (Q_vx is not None) or (Q_epsi is not None) or (Q_ey is not None):
                    config_data.set(
                        "cbf_cost_evaluator",
                        "Q",
                        f"[{Q_vx}, 3.0, 0.1, 0.0, 0.0, {Q_epsi}, {Q_ey}, 0.0]",
                    )

                    config_data.set(
                        "baseline_cost_evaluator",
                        "Q",
                        f"[{Q_vx}, 3.0, 0.1, 0.0, 0.0, {Q_epsi}, {Q_ey}, 0.0]",
                    )

                config_data.set(traj_sampler_name, "number_of_trajectories", str(n_traj))

                if test_agent == "CEM":
                    config_data.set(
                        f"CEM_controller",
                        "control_horizon",
                        str(horizon),
                    )
                else:
                    config_data.set(
                        f"{test_agent}_mppi_controller",
                        "control_horizon",
                        str(horizon),
                    )
                if test_agent == "cbf" or test_agent == "shield_risk_aware":
                    config_data.set(
                        "cbf_cost_evaluator",
                        "cbf_alpha",
                        str(alpha),
                    )

                if render_png:
                    config_data.set(
                        "renderer1",
                        "figure_size",
                        "[20, 20]",
                    )
                    # self.figure_size = np.asarray(ast.literal_eval(config_data.get(section_name, "figure_size")),
                    #                               dtype=int)

                agent: SimulatedRobot = factory_from_config(
                    robot_factory_base, config_data, test_agent + "_agent"
                )


                # # Reset agent to near crash timestep
                # crash_states = np.load('crash_traj.npz')['sim_states'].T
                # agent.reset_state(crash_states[-13][:8])
                # agent.reset_controller()

                renderer1 = factory_from_config(
                    renderer_factory_base, config_data, "renderer1"
                )
                logger = factory_from_config(logger_factory_base, config_data, "logger")
                assert isinstance(logger, AutorallyNpzLogger)
                logger.set_agent(agent=agent)
                if render:
                    agent.set_renderer(renderer=renderer1)
                goal_checker_for_checking_vehicle_position = factory_from_config(
                    goal_checker_factory_base,
                    config_data,
                    "my_goal_checker_for_checking_vehicle_position",
                )
                collision_checker_for_failure = factory_from_config(
                    collision_checker_factory_base,
                    config_data,
                    "my_collision_checker_for_crash"
                )
                steps = 0
                total_eval_time = 0.0
                eval_times = deque([], maxlen=10)

                nominal_dynamics = factory_from_config(dynamics_factory_base, config_data, "sim_dynamics1")

                controller: MPPI = agent.controller
                evaluator: AutorallyMPPICostEvaluator = controller.cost_evaluator
                goal_checker: AutorallyCartesianGoalChecker = evaluator.goal_checker


                # evaluator.collision_cost = 1e6
                # # evaluator.collision_cost = 1.3e3
                # print(f"{evaluator.collision_cost=}")

                tgt_vel_orig = goal_checker.goal_state[0]
                if tgt_vel is not None:
                    loguru.logger.info("Overriding the tgt_vel from {} -> {}!".format(tgt_vel_orig, tgt_vel))
                    goal_checker.goal_state[0] = tgt_vel
                else:
                    loguru.logger.info("tgt_vel: {}".format(tgt_vel_orig))

                # if not np.allclose(np.diag(evaluator.Q), np.array([50.0, 3.0, 0.1, 0.0, 0.0, 0.0, 30.0, 0.0])):
                #     raise ValueError("Q is not correct!")
                # if not np.allclose(np.diag(evaluator.QN), np.array([50.0, 100.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0])):
                #     raise ValueError("QN is not correct!")

                max_timesteps = 1500
                timesteps = 0

                # Run Autorally
                ctrl_times = []


                if render_png:
                    png_traj_path = f"{ablation}_traj_png"
                    if os.path.exists(png_traj_path):
                        shutil.rmtree(png_traj_path, ignore_errors=True)
                    os.makedirs(png_traj_path)


                    renderer1.render_goal(
                        goal_checker_for_checking_vehicle_position.get_goal()
                    )
                    fig, ax = agent.render_all(plot=True)
                    fig.savefig(f"{png_traj_path}/{timesteps}.png")
                    plt.close(fig)

                collisions = 0
                while not logger.goal_checker.check(agent.state.reshape((-1, 1))):
                    rng = onp.random.default_rng(seed=12345)

                    timer = Timer("Control loop").start()
                    timer_ = timer.child("render").start()
                    if render:
                        renderer1.render_goal(
                            goal_checker_for_checking_vehicle_position.get_goal()
                        )
                    steps += 1
                    timer_.stop()
                    timer_ = timer.child("take_action_with_controller").start()
                    old_state = agent.state
                    state_next, cost, eval_time, action = agent.take_action_with_controller(return_time=True, logger=logger)
                    timer_.stop()

                    # # Keep track of how long each control took
                    # ctrl_times.append(float(timer._children['take_action_with_controller'].elapsed))

                    timer_ = timer.child("calc_agent_disturb").start()
                    # print(logger.disturbances.shape, logger.sim_states.shape)
                    logger.calculate_agent_disturbance(state_next, old_state, action, nominal_dynamics)
                    timer_.stop()

                    eval_times.append(eval_time)
                    timer_ = timer.child("calc laps").start()
                    logger.calculate_number_of_laps(
                        state_next,
                        dynamics=agent.dynamics,
                        goal_checker=goal_checker_for_checking_vehicle_position,
                    )
                    timer_.stop()
                    timer_ = timer.child("calc col").start()
                    logger.calculate_number_of_collisions(
                        state_next,
                        dynamics=agent.dynamics,
                        collision_checker=agent.cost_evaluator.collision_checker,
                    )
                    timer_.stop()
                    timer_ = timer.child("calc failures").start()
                    logger.calculate_number_of_failures(
                        state_next,
                        dynamics=agent.dynamics,
                        collision_checker=collision_checker_for_failure)
                    timer_.stop()
                    timer_ = timer.child("logger.log").start()
                    logger.log()
                    timer_.stop()
                    if collisions != logger.number_of_collisions:
                        collisions = logger.number_of_collisions
                        loguru.logger.critical(f"COLLIDED!!! # collisions: {collisions}")

                    if logger.crash == 1:

                        # # Save logger to npz somewhere
                        # with open(f"crash_traj.npz", "wb") as f:
                        #     np.savez(f, sim_states=logger.sim_states)
                        #     # logger.sim_states


                        if render_png:
                            renderer1.render_goal(
                                goal_checker_for_checking_vehicle_position.get_goal()
                            )
                            # fig, ax = agent.render_all(plot=True)
                            fig.savefig(f"{png_traj_path}/{timesteps}.png")
                            plt.close(fig)

                        loguru.logger.critical("VEHICLE CRASHED!!!!")
                        break
                    # print("Average eval time: ", sum(eval_times) / len(eval_times), "Control update rate: ", 1/(sum(eval_times) / len(eval_times)))
                    timer_.stop()
                    timer.stop().print_results()

                    # Early stopping if we are at a local minimum
                    timesteps += 1
                    if timesteps >= max_timesteps:
                        loguru.logger.debug(f'Early stopping due to max timesteps reached!')
                        logger.increment_timeouts()
                        break


                    if render_png:
                        renderer1.render_goal(
                            goal_checker_for_checking_vehicle_position.get_goal()
                        )
                        fig, ax = agent.render_all(plot=True)
                        fig.savefig(f"{png_traj_path}/{timesteps}.png")
                        plt.close(fig)

                # Update control hertz
                # logger.set_ctrl_hz(1/np.mean(np.array(ctrl_times[1:])))


                # # Save states of logger
                # with open(f"{ablation}_lr_exp_states.npz", "wb") as f:
                #     np.savez(f, sim_states=logger.sim_states)

                # Update the final number of laps and save the log
                logger.calculate_number_of_laps(
                    state_next,
                    dynamics=agent.dynamics,
                    goal_checker=agent.cost_evaluator.goal_checker,
                )
                if log:
                    logger.shutdown(mean_eval_time=total_eval_time / steps)
                print(f"Mean controller eval time: {total_eval_time / steps}")
                print(f"Collision number: {logger.number_of_collisions}")
                renderer1.close()


if __name__ == "__main__":
    import time
    start_time = time.time()
    with ipdb.launch_ipdb_on_exception():
        test = TestRunAutorallyMPPI()
        test.test_run_autorally_MPPI()

    print(f'Time taken: {time.time() - start_time}')

