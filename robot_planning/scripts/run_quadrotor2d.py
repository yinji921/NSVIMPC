try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
from robot_planning.environment.robots.simulated_robot import SimulatedRobot
from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factories import renderer_factory_base
from robot_planning.factory.factories import logger_factory_base
from robot_planning.factory.factories import goal_checker_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.helper.timer import Timer
import jax.numpy as np
import time
import os
import jax
from jax import profiler
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
jax.config.update('jax_platform_name', 'gpu')
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)


def main():
    # agent_name = "baseline"
    # agent_name = "cbf"
    agent_name = "ncbf"

    config_path = "configs/run_quadrotor2d.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(config_path)
    agent1 = factory_from_config(robot_factory_base, config_data, agent_name+'_agent')
    render = True

    goal_checker_for_checking_quadrotor_position = factory_from_config(
        goal_checker_factory_base,
        config_data,
        "my_goal_checker1",
    )

    if render:
        renderer1 = factory_from_config(renderer_factory_base, config_data, 'renderer1')
        agent1.set_renderer(renderer=renderer1)
        renderer1.render_goal(goal_checker_for_checking_quadrotor_position.get_goal())

    while not (agent1.cost_evaluator.goal_checker.check(agent1.state.reshape((-1, 1))) ):
    # while True:
        timer = Timer("Control loop").start()
        state_next_1, cost_1, eval_time_1, action_1 = agent1.take_action_with_controller(return_time=True)
        print("agent 1 current state: ", state_next_1)
        # timer.stop().print_results()
        timer.stop().print_control_freq_result()

        if agent1.controller.cost_evaluator.collision_checker.check(state_next_1):
            print("Collision detected!")
            break



if __name__ == '__main__':
    main()
