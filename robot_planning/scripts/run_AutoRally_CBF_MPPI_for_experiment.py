import jax.numpy as np
import jax
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
# jax.config.update('jax_platform_name', 'gpu')
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

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
        # Parameter set up
        experiment_ids = range(1)
        n_trajectories_setting = [1000]
        control_horizon_setting = [20]
        cbf_alpha_settings = [0.9]
        # Q_epsi_setting = [10.0]
        # Q_ey_setting = [40.0]  # also try values 30.0 to 50.0
        Q_vx = None
        Q_epsi_setting = [None]
        Q_ey_setting = [None]  # also try values 30.0 to 50.0

        tgt_vel = None
        # tgt_vel = 5.0

        render = True
        log = False
        test_agents = ["ncbf"]
        np.set_printoptions(
            edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%10.3f" % x)
        )
        # config_path = "configs/run_Autorally_CBF_MPPI_for_experiment.cfg"
        config_path = "configs/reverse_run_Autorally_CBF_MPPI_for_experiment.cfg"
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
        )
        for test_agent, experiment_index, n_traj, horizon, alpha, Q_epsi, Q_ey in experiments:
            print(f"{test_agent}: {(experiment_index, n_traj, horizon, alpha, Q_epsi, Q_ey)}")

            # Use the experiment_index as the seed.
            seed = 12345 + experiment_index
            traj_sampler_name = f"{test_agent}_stochastic_trajectories_sampler"
            noise_sampler_name = config_data.get(traj_sampler_name, "noise_sampler")
            config_data.set(noise_sampler_name, "seed", str(seed))

            # Update the batch code and experiment parameters
            config_data.set("logger", "batch_code", str(experiment_index))
            config_data.set(
                "logger",
                "experiment_name",
                f"{test_agent}_n_traj_{n_traj}_horizon_{horizon}_alpha_{alpha}_Qepsi_{Q_epsi}_Qey_{Q_ey}",
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

            agent: SimulatedRobot = factory_from_config(
                robot_factory_base, config_data, test_agent + "_agent"
            )
            renderer1 = factory_from_config(
                renderer_factory_base, config_data, "renderer1"
            )
            # logger = factory_from_config(logger_factory_base, config_data, "logger")
            # assert isinstance(logger, AutorallyNpzLogger)
            # logger.set_agent(agent=agent)
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
            total_eval_time = 0.0
            eval_times = deque([], maxlen=10)

            nominal_dynamics = factory_from_config(dynamics_factory_base, config_data, "sim_dynamics1")

            controller: MPPI = agent.controller
            evaluator: AutorallyMPPICostEvaluator = controller.cost_evaluator
            goal_checker: AutorallyCartesianGoalChecker = evaluator.goal_checker

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

            # Run Autorally
            # while not logger.goal_checker.check(agent.state.reshape((-1, 1))):
            rng = onp.random.default_rng(seed=12345)
            for ii in range(10):
                # [ vx, vy, wz, wF, wR, psi, X, Y ]
                start_state = onp.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.14, 0.0, -2.0])
                # start_state = onp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0])
                steps = 0
                accumulated_steps = 0
                number_of_laps = 0
                if ii > 0:
                    # # Sample a random s, and start from there.
                    # track = agent.controller.dynamics.track
                    # s = rng.uniform(0.0, track.s_total)
                    # x0, y0, theta0, _, _ = track.get_lerp_from_s(s)
                    # start_state[-3:] = np.array([theta0, x0, y0])

                    # Add some noise.
                    start_state[-2:] += rng.normal(0.0, 0.3, 2)

                agent.reset_state(np.array(start_state))
                agent.reset_controller()

                logger = factory_from_config(logger_factory_base, config_data, "logger")
                assert isinstance(logger, AutorallyNpzLogger)
                logger.set_agent(agent=agent)

                while logger.number_of_laps < 10:
                    if logger.number_of_laps > number_of_laps:
                        print("lap time = ", steps * agent.dynamics.delta_t, " Average lap time = ", accumulated_steps * agent.dynamics.delta_t/logger.number_of_laps)
                        steps = 0
                        number_of_laps = logger.number_of_laps
                    timer = Timer("Control loop").start()
                    timer_ = timer.child("render").start()
                    if render:
                        renderer1.render_goal(
                            goal_checker_for_checking_vehicle_position.get_goal()
                        )
                    timer_.stop()
                    timer_ = timer.child("take_action_with_controller").start()
                    old_state = agent.state
                    state_next, cost, eval_time, action = agent.take_action_with_controller(return_time=True)
                    timer_.stop()
                    timer_ = timer.child("calc_agent_disturb").start()
                    # print(logger.disturbances.shape, logger.sim_states.shape)
                    logger.calculate_agent_disturbance(state_next, old_state, action, nominal_dynamics)
                    timer_.stop()

                    eval_times.append(eval_time)
                    steps += 1
                    accumulated_steps += 1
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
                    if logger.crash == 1:
                        print("vehicle crashed!")
                        break
                    # print("Average eval time: ", sum(eval_times) / len(eval_times), "Control update rate: ", 1/(sum(eval_times) / len(eval_times)))
                    timer_.stop()
                    timer.stop().print_results()


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
            # renderer1.close()

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        test = TestRunAutorallyMPPI()
        test.test_run_autorally_MPPI()
