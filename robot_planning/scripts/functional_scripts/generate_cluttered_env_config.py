import numpy as np
from tqdm import tqdm
aa = np.arange(5)
import numpy as onp
from robot_planning.environment.robots.simulated_robot import SimulatedRobot
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
import os
from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
import ipdb


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
# jax.config.update('jax_platform_name', 'gpu')
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)




rng = onp.random.default_rng(seed=12345)
test_agent = "baseline"
number_of_obstacles = 10
obstacle_position_variance = 0.3
obstacle_radius_mean = 0.8
obstacle_radius_variance = 0.2
config_path = "configs/reverse_run_Autorally_CBF_MPPI_for_experiment.cfg"
config_data = ConfigParser.ConfigParser()
config_data.read(config_path)
agent: SimulatedRobot = factory_from_config(robot_factory_base, config_data, test_agent + "_agent")

track = agent.controller.dynamics.track
obstacles = []
obstacles_radius = []
for i in tqdm(range(number_of_obstacles)):
    s = rng.uniform(0.0, track.s_total)
    x0, y0, theta0, _, _ = track.get_lerp_from_s(s)
    obstacle = np.array([[x0, y0]]).squeeze()
    obstacle += rng.normal(0.0, obstacle_position_variance, 2)
    obstacle_radius = abs(rng.normal(obstacle_radius_mean, obstacle_radius_variance))

    # convert obstacle to list
    obstacle = obstacle.tolist()
    obstacles.append(obstacle)
    obstacles_radius.append(obstacle_radius)

config_data.set("my_collision_checker_for_collision", "obstacles", str(obstacles))
config_data.set("my_collision_checker_for_collision", "obstacles_radius", str(obstacles_radius))
config_data.set("my_collision_checker_for_crash", "obstacles", str(obstacles))
config_data.set("my_collision_checker_for_crash", "obstacles_radius", str(obstacles_radius))

with open("configs/" + "reverse_run_Autorally_cluttered_env" + ".cfg", "w") as configfile:
    config_data.write(configfile)
    configfile.close()


