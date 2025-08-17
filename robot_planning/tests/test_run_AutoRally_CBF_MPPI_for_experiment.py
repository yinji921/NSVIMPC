try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
import os
import jax.numpy as np
import numpy as onp
from robot_planning.environment.robots.simulated_robot import SimulatedRobot
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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress CUDA warnings

class TestRunAutorallyMPPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")

    def test_run_autorally_MPPI(self):
        """Test all AutoRally methods: baseline, cbf, cbf_inefficient, ncbf"""
        # Simple test configuration
        render = False  # Disable rendering for testing
        log = False
        test_agents = ["baseline", "cbf", "cbf_inefficient", "ncbf"]  # Test all methods
        
        # Single test parameters
        n_traj = 50  # Reduced for testing
        horizon = 5   # Reduced for testing
        alpha = 0.9
        
        np.set_printoptions(
            edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%10.3f" % x)
        )
        
        config_path = "configs/test_reverse_run_Autorally_CBF_MPPI_for_experiment.cfg"
        config_data = ConfigParser.ConfigParser()
        config_data.read(config_path)

        for test_agent in test_agents:
            print(f"\n=== Testing {test_agent} ===")
            print(f"Testing {test_agent}: n_traj={n_traj}, horizon={horizon}, alpha={alpha}")

            # Set seed for reproducibility
            seed = 12345
            traj_sampler_name = f"{test_agent}_stochastic_trajectories_sampler"
            noise_sampler_name = config_data.get(traj_sampler_name, "noise_sampler")
            config_data.set(noise_sampler_name, "seed", str(seed))

            # Update the configuration parameters
            config_data.set("logger", "batch_code", "0")
            config_data.set(
                "logger",
                "experiment_name",
                f"test_{test_agent}_n_traj_{n_traj}_horizon_{horizon}_alpha_{alpha}",
            )

            # Update trajectory and horizon settings
            config_data.set(traj_sampler_name, "number_of_trajectories", str(n_traj))
            config_data.set(
                f"{test_agent}_mppi_controller",
                "control_horizon",
                str(horizon),
            )

            # Create agent and components
            agent = factory_from_config(
                robot_factory_base, config_data, test_agent + "_agent"
            )
            renderer1 = factory_from_config(
                renderer_factory_base, config_data, "renderer1"
            )
            logger = factory_from_config(logger_factory_base, config_data, "logger")
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
            
            nominal_dynamics = factory_from_config(dynamics_factory_base, config_data, "sim_dynamics1")

            # Simple test run - just one iteration
            start_state = onp.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.14, 0.0, -2.0])
            agent.reset_state(np.array(start_state))
            agent.reset_controller()

            # Run limited simulation steps
            steps = 0
            max_steps = 3  # Very limited for testing

            # Run AutoRally
            while not logger.goal_checker.check(agent.state.reshape((-1, 1))) and steps < max_steps:
                if render:
                    renderer1.render_goal(
                        goal_checker_for_checking_vehicle_position.get_goal()
                    )
                
                old_state = agent.state
                state_next, cost, eval_time, action = agent.take_action_with_controller(return_time=True)
                
                # Calculate logging metrics
                logger.calculate_agent_disturbance(state_next, old_state, action, nominal_dynamics)
                logger.calculate_number_of_laps(
                    state_next,
                    dynamics=agent.dynamics,
                    goal_checker=goal_checker_for_checking_vehicle_position,
                )
                logger.calculate_number_of_collisions(
                    state_next,
                    dynamics=agent.dynamics,
                    collision_checker=agent.cost_evaluator.collision_checker,
                )
                logger.calculate_number_of_failures(
                    state_next,
                    dynamics=agent.dynamics,
                    collision_checker=collision_checker_for_failure)
                logger.log()
                
                steps += 1
                
                if logger.crash == 1:
                    print("vehicle crashed!")
                    break
                    
                print(f"Step {steps}: Eval time: {eval_time:.4f}s")

            # Finalize the test
            logger.calculate_number_of_laps(
                agent.state if steps == 0 else state_next,
                dynamics=agent.dynamics,
                goal_checker=agent.cost_evaluator.goal_checker,
            )
            
            if steps > 0:
                print(f"Test completed! Steps: {steps}")
                print(f"Collision number: {logger.number_of_collisions}")
            else:
                print("Test completed with no steps")
                
            print(f"{test_agent} test passed!")
            
        print("\nAll AutoRally CBF MPPI experiment tests passed!")


if __name__ == "__main__":
    test = TestRunAutorallyMPPI()
    test.test_run_autorally_MPPI()