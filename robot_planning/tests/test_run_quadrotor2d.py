import unittest
import os
import sys
import jax
import jax.numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress CUDA warnings

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import robot_factory_base, renderer_factory_base, goal_checker_factory_base
from robot_planning.helper.timer import Timer


class TestRunQuadrotor2D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Setting up TestRunQuadrotor2D")
        
    @classmethod
    def tearDownClass(cls):
        print("Tearing down TestRunQuadrotor2D")
        
    def test_run_quadrotor2d_all_methods(self):
        """Test all Quadrotor2D methods: baseline, cbf, cbf_inefficient, ncbf"""
        print("Testing all Quadrotor2D methods...")
        
        render = False  # Disable rendering for testing
        test_agents = ["baseline", "cbf", "cbf_inefficient", "ncbf"]
        
        config_path = "configs/test_run_quadrotor2d.cfg"
        config_data = ConfigParser.ConfigParser()
        config_data.read(config_path)

        for agent_name in test_agents:
            print(f"\n=== Testing Quadrotor2D {agent_name} ===")
            
            # Create agent
            agent = factory_from_config(robot_factory_base, config_data, agent_name + '_agent')
            
            # Create goal checker
            goal_checker_for_checking_position = factory_from_config(
                goal_checker_factory_base,
                config_data,
                "my_goal_checker1",
            )

            # Setup renderer if needed
            if render:
                renderer = factory_from_config(renderer_factory_base, config_data, 'renderer1')
                agent.set_renderer(renderer=renderer)
                renderer.render_goal(goal_checker_for_checking_position.get_goal())

            # Run limited simulation steps
            steps = 0
            max_steps = 3  # Very limited for testing

            print(f"Initial state: {agent.state}")

            # Run simulation
            while not (agent.cost_evaluator.goal_checker.check(agent.state.reshape((-1, 1)))) and steps < max_steps:
                timer = Timer("Control loop").start()
                state_next, cost, eval_time, action = agent.take_action_with_controller(return_time=True)
                timer.stop()
                
                print(f"Step {steps + 1}: action={action}, state={state_next}, eval_time={eval_time:.4f}s")
                
                steps += 1

                # Check for collision
                if agent.controller.cost_evaluator.collision_checker.check(state_next):
                    print("Collision detected!")
                    break

            print(f"{agent_name} test completed! Steps: {steps}")
            print(f"Final state: {agent.state}")
            
            # Basic assertion - agent should have executed at least one step
            self.assertGreaterEqual(steps, 1, f"{agent_name} agent should execute at least one control step")
            
        print("\nAll Quadrotor2D tests passed!")


if __name__ == "__main__":
    unittest.main()