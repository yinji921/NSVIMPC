import jax.numpy as np
aa = np.asarray(5)
import configparser as ConfigParser
import pathlib
import pickle

import ipdb
from robot_planning.controllers.MPPI.MPPI import MPPI
from robot_planning.environment.cost_evaluators import AutorallyMPPICostEvaluator
from robot_planning.environment.goal_checker import AutorallyCartesianGoalChecker


import loguru
import numpy as onp

from robot_planning.batch_experimentation.loggers import AutorallyNpzLogger
from robot_planning.environment.robots.simulated_robot import SimulatedRobot
from robot_planning.factory.factories import (collision_checker_factory_base, goal_checker_factory_base,
                                              logger_factory_base, renderer_factory_base, robot_factory_base)
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.helper.timer import Timer


def set_config(config_data, test_agent, n_traj, control_horizon, cbf_alpha, experiment_index):
    traj_sampler_name = f"{test_agent}_stochastic_trajectories_sampler"
    noise_sampler_name = config_data.get(traj_sampler_name, "noise_sampler")

    seed = 12345 + experiment_index
    config_data.set(noise_sampler_name, "seed", str(seed))

    # Update the batch code and experiment parameters
    config_data.set("logger", "batch_code", str(experiment_index))

    config_data.set(traj_sampler_name, "number_of_trajectories", str(n_traj))

    config_data.set(
        f"{test_agent}_mppi_controller",
        "control_horizon",
        str(control_horizon),
    )


def main():
    clock_wise = False  # False for counter clock-wise
    render = True
    if clock_wise is False:
        ## for counter clock-wise
        # config_path = "configs/reverse_run_Autorally_CBF_MPPI_for_experiment.cfg"
        config_path = "configs/reverse_run_Autorally_cluttered_env.cfg"
    else:
        ## for clock-wise
        config_path = "configs/run_Autorally_CBF_MPPI_for_experiment.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(config_path)

    test_agent = "baseline"
    experiment_index = 0
    n_traj = 1_000
    control_horizon = 20
    cbf_alpha = 0.9

    tgt_vel = None
    # n_trajs_collect = 128
    n_trajs_collect = 384

    # tgt_vel = 5.0
    # n_trajs_collect = 32

    n_laps_log = 1

    set_config(config_data, test_agent, n_traj, control_horizon, cbf_alpha, experiment_index)

    agent: SimulatedRobot = factory_from_config(robot_factory_base, config_data, test_agent + "_agent")
    renderer1 = factory_from_config(renderer_factory_base, config_data, "renderer1")
    if render:
        agent.set_renderer(renderer=renderer1)

    # print("tracking velocity: ", agent.controller.cost_evaluator.goal_checker.goal_state[0])
    if tgt_vel is not None:
        controller: MPPI = agent.controller
        evaluator: AutorallyMPPICostEvaluator = controller.cost_evaluator
        goal_checker: AutorallyCartesianGoalChecker = evaluator.goal_checker
        tgt_vel_orig = goal_checker.goal_state[0]
        loguru.logger.info("Overriding the tgt_vel from {} -> {}!".format(tgt_vel_orig, tgt_vel))
        goal_checker.goal_state[0] = tgt_vel

    goal_checker_for_checking_vehicle_position = factory_from_config(
        goal_checker_factory_base,
        config_data,
        "my_goal_checker_for_checking_vehicle_position",
    )
    collision_checker_for_failure = factory_from_config(
        collision_checker_factory_base, config_data, "my_collision_checker_for_crash"
    )

    bT_x = []

    rng = onp.random.default_rng(seed=12345)
    for traj_idx in range(n_trajs_collect):
        print("               ===== {:3} / {:3} =====            ".format(traj_idx + 1, n_trajs_collect))
        # [ vx, vy, wz, wF, wR, psi, X, Y ]
        if clock_wise is False:
            ## for reversed (counter clock-wise)
            start_state = onp.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.14, 0.0, -2.8])
        else:
            ## clock-wise
            start_state = onp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.8])
        if traj_idx > 0:
            # Sample a random s, and start from there.
            track = agent.controller.dynamics.track
            s = rng.uniform(0.0, track.s_total)
            x0, y0, theta0, _, _ = track.get_lerp_from_s(s)
            if clock_wise is False:
                start_state[-3:] = np.array([theta0 + np.pi, x0, y0])
            else:
                start_state[-3:] = np.array([theta0, x0, y0])
            # Add some noise.
            start_state[-2:] += rng.normal(0.0, 0.3, 2)
            start_state[-2] += rng.normal(0.0, 0.2)

        agent.reset_state(np.array(start_state))
        agent.reset_controller()
        logger = factory_from_config(logger_factory_base, config_data, "logger")
        assert isinstance(logger, AutorallyNpzLogger)
        logger.set_agent(agent=agent)

        renderer1.render_goal(goal_checker_for_checking_vehicle_position.get_goal())
        while logger.number_of_laps < n_laps_log:
            timer = Timer("Control loop").start()
            state_next, cost, eval_time, action = agent.take_action_with_controller(return_time=True)

            logger.calculate_number_of_laps(
                state_next,
                dynamics=agent.dynamics,
                goal_checker=logger.goal_checker,
            )
            logger.calculate_number_of_collisions(
                state_next,
                dynamics=agent.dynamics,
                collision_checker=agent.cost_evaluator.collision_checker,
            )
            logger.calculate_number_of_failures(
                state_next, dynamics=agent.dynamics, collision_checker=collision_checker_for_failure
            )
            logger.log()

            timer.stop().print_results()
            if logger.crash == 1:
                print("vehicle crashed at kk={}!".format(logger.n_sim_states))
                break

        # Pull out the state trajs.
        T_x = logger.sim_states
        bT_x.append(T_x)

    # Save the trajectories.
    data_dir = pathlib.Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    pkl_path = data_dir / "raw_data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(bT_x, f, pickle.HIGHEST_PROTOCOL)
    loguru.logger.info("Saved to {}!".format(pkl_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
