import time

import jax.numpy as np
import tqdm

from robot_planning.environment.collision_checker import Quadrotor2DCollisionChecker

aa = np.asarray(5)
import configparser as ConfigParser
import pathlib
import pickle

import ipdb
import loguru
import numpy as onp

from robot_planning.batch_experimentation.loggers import AutorallyNpzLogger, Drone2DNpzLogger
from robot_planning.controllers.MPPI.MPPI import MPPI
from robot_planning.environment.cost_evaluators import AutorallyMPPICostEvaluator, Quadrotor2DCBFCostEvaluator
from robot_planning.environment.goal_checker import AutorallyCartesianGoalChecker, QuadrotorCartesianGoalChecker
from robot_planning.environment.robots.simulated_robot import SimulatedRobot
from robot_planning.factory.factories import (collision_checker_factory_base, goal_checker_factory_base,
                                              logger_factory_base, renderer_factory_base, robot_factory_base)
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.helper.timer import Timer


def main():
    config_path = "configs/run_quadrotor2d.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(config_path)

    # n_trajs_collect = 256
    # n_trajs_collect = 1_024
    n_trajs_collect = 8_192

    render = False

    n_traj = 1_000
    horizon = 20
    tgt_vx = 8.0
    # tgt_vx = 7.0
    Q_x = 0.0
    Q_vx = 10.0

    test_agent = "cbf"

    seed = 12345 + 0
    traj_sampler_name = f"{test_agent}_stochastic_trajectories_sampler"
    noise_sampler_name = config_data.get(traj_sampler_name, "noise_sampler")
    config_data.set(noise_sampler_name, "seed", str(seed))

    if (Q_vx is not None) or (Q_x is not None):
        config_data.set(
            "cbf_cost_evaluator",
            "Q",
            f"[{Q_x}, 30.0, 50.0, {Q_vx}, 10.0, 10.0]",
        )
        config_data.set(
            "baseline_cost_evaluator",
            "Q",
            f"[{Q_x}, 30.0, 50.0, {Q_vx}, 10.0, 10.0]",
        )

    traj_sampler_name = f"{test_agent}_stochastic_trajectories_sampler"
    config_data.set(traj_sampler_name, "number_of_trajectories", str(n_traj))

    config_data.set(
        f"{test_agent}_mppi_controller",
        "control_horizon",
        str(horizon),
    )

    agent: SimulatedRobot = factory_from_config(robot_factory_base, config_data, test_agent + "_agent")
    renderer1 = factory_from_config(renderer_factory_base, config_data, "renderer1")

    controller: MPPI = agent.controller
    evaluator: Quadrotor2DCBFCostEvaluator = controller.cost_evaluator
    goal_checker: QuadrotorCartesianGoalChecker = evaluator.goal_checker

    if tgt_vx is not None:
        tgt_vel_orig = goal_checker.goal_state[3]
        loguru.logger.critical("Overriding the tgt_vx from {} -> {}!".format(tgt_vel_orig, tgt_vx))
        goal_checker.goal_state[3] = tgt_vx

    collision_checker_for_failure: Quadrotor2DCollisionChecker = factory_from_config(
        collision_checker_factory_base, config_data, "my_collision_checker_for_collision"
    )

    # -------------------------------------------------------------------------
    #  Start collecting data.
    # -------------------------------------------------------------------------
    T_x = None
    bT_x = []

    prev_status = None

    rng = onp.random.default_rng(seed=12345)
    for traj_idx in tqdm.trange(n_trajs_collect):
        # print("               ===== {:3} / {:3} =====            ".format(traj_idx + 1, n_trajs_collect))
        start_state = onp.array([-4, 1.0, 0.0, 0.0, 0.0, 0.0])

        if traj_idx > 0:
            assert prev_status is not None

            if prev_status == "success":
                # Probability of sampling from the narrow corridor.
                p_corridor = 0.5

                if p_corridor:
                    # Get all the indices where -1 <= px <= 3.
                    kk_corridor = onp.where((T_x[0, :] >= -1) & (T_x[0, :] <= 3))[0]

                    if len(kk_corridor) > 0:
                        kk_random = rng.choice(kk_corridor)
                    else:
                        loguru.logger.warning("No corridor samples found. Sampling randomly.")
                        kk_random = rng.integers(0, len(T_x))

                    # Use a tiny position perturbation.
                    noise_std = onp.array([0.001, 0.001, 0.001, 0.5, 0.5, 0.5])
                else:
                    # Just sample randomly.
                    n_from_end = 20

                    lo = 0
                    hi = max(1, len(T_x) - n_from_end)
                    kk_random = rng.integers(lo, hi)
                    noise_std = onp.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])

            elif prev_status == "crash":
                n_from_end = 5

                lo = 0
                hi = max(1, len(T_x) - n_from_end)
                kk_random = rng.integers(lo, hi)
                noise_std = onp.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
            else:
                raise ValueError(f"Unknown prev_status: {prev_status}")

            p_start_prev = 0.2
            start_prev = rng.binomial(1, p_start_prev)

            # Don't start_prev if the previous traj was a crash and it was very short
            dont_start_prev = (prev_status == "crash") and (len(T_x) < 10)
            start_prev = start_prev and (not dont_start_prev)

            if start_prev:
                loguru.logger.info("start prev")
                # Sample a random state from the previous traj and perturb it.

                # (nx, )
                x = T_x[:, kk_random]
            else:
                loguru.logger.info("start random")

                buffer = np.deg2rad(2)
                lo = onp.array([-4.5, 0.2, -np.pi / 2 + buffer, -1.0, -0.5, 1.0])
                hi = onp.array([-1.5, 1.0, np.pi / 2 - buffer, 9.0, 0.5, 1.0])

                # lo = onp.array([2.0, 0.2, -np.pi / 2 + buffer, -1.0, -0.5, 1.0])
                # hi = onp.array([4.0, 1.0, np.pi / 2 - buffer, 3.0, 0.5, 1.0])

                x = rng.uniform(lo, hi)
                invalid_x0 = False
                while invalid_x0:
                    x = rng.uniform(lo, hi)
                    invalid_x0 = collision_checker_for_failure.check(x)
                print("out of loop")

            if start_prev:
                # start_state = x
                # Add some noise.
                has_collided = True
                print("in loop2")
                while has_collided:
                    start_state = x + rng.normal(0.0, noise_std)
                    has_collided = collision_checker_for_failure.check(start_state)
                print("out of loop2")

                # With some probability, scale the velocity with random multipliers.
                p_scale_vel = 0.1
                scale_vel = rng.binomial(1, p_scale_vel)
                if scale_vel:
                    vel_scale = rng.uniform(0.9, 1.1, size=(2,))
                    start_state[3:5] *= vel_scale

                    # clip the velocity
                    start_state[3:5] = onp.clip(start_state[3:5], -10.0, 10.0)
            else:
                start_state = x

        agent.reset_state(np.array(start_state))
        agent.reset_controller()
        logger = factory_from_config(logger_factory_base, config_data, "logger")
        assert isinstance(logger, Drone2DNpzLogger)
        logger.set_agent(agent=agent)
        if render:
            agent.set_renderer(renderer=renderer1)

        goal_checker_for_checking_drone_position: QuadrotorCartesianGoalChecker = factory_from_config(
            goal_checker_factory_base,
            config_data,
            "my_goal_checker1",
        )
        steps = 0

        logger.append_state()

        if render:
            renderer1.render_goal(goal_checker_for_checking_drone_position.get_goal())

        print("start control loop")
        reached_goal = False
        while not reached_goal:
            timer = Timer("Control loop").start()

            steps += 1
            state_next, cost, eval_time, action = agent.take_action_with_controller(return_time=True, logger=logger)
            logger.calculate_number_of_failures(state_next, collision_checker_for_failure)
            logger.log()

            if render:
                renderer1.render_goal(goal_checker_for_checking_drone_position.get_goal())

            timer.stop().print_results()

            if logger.crash == 1:
                loguru.logger.critical("CRASH!!!!")
                prev_status = "crash"
                # ipdb.set_trace()
                break

            if steps > 100:
                loguru.logger.critical("TIMEOUT!!!!")
                prev_status = "crash"
                break

            reached_goal = logger.goal_checker.check(agent.state.reshape((-1, 1)))

        if reached_goal:
            prev_status = "success"

        # Pull out the state trajs.
        # (nx, T). nx = 6, [x, z, theta, vx, vz, omega]
        T_x = logger.sim_states

        if onp.isnan(T_x).any():
            loguru.logger.critical("NaN detected!")
            ipdb.set_trace()

        bT_x.append(T_x)

    # Save the trajectories.
    data_dir = pathlib.Path(__file__).parent.parent.parent / "data_drone"
    data_dir.mkdir(exist_ok=True, parents=True)

    pkl_path = data_dir / "raw_data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(bT_x, f, pickle.HIGHEST_PROTOCOL)
    loguru.logger.info("Saved to {}!".format(pkl_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
