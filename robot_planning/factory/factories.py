def dynamics_factory_base(base_type):
    # if base_type == "bicycle_dynamics":
    #     from robot_planning.environment.dynamics.bicycle_dynamics import BicycleDynamics
    #     return BicycleDynamics()
    # elif base_type == "abstract_dynamics":
    #     from robot_planning.environment.dynamics.abstract_dynamics import (
    #         AbstractDynamics,
    #     )
    #     return AbstractDynamics()
    if (
        base_type == "autorally_dynamics_cartesian"
        or base_type == "autorally_dynamics_map"
    ):
        from robot_planning.environment.dynamics.autorally_dynamics.autorally_dynamics import (
            AutoRallyDynamics,
        )

        return AutoRallyDynamics(dynamics_type=base_type)
    elif base_type == "autorally_dynamics_clbf":
        from robot_planning.environment.dynamics.autorally_dynamics.autorally_surrogate_dynamics import (
            ControlAffineAutoRallyDynamics,
        )

        return ControlAffineAutoRallyDynamics(dynamics_type=base_type)
    elif base_type == "point_dynamics":
        from robot_planning.environment.dynamics.point_dynamics import PointDynamics
        return PointDynamics()
    elif base_type == 'bicycle_dynamics':
        from robot_planning.environment.dynamics.bicycle_dynamics import BicycleDynamics
        return BicycleDynamics()
    elif base_type == "quadrotor_dynamics_2d":
        from robot_planning.environment.dynamics.quadrotor2d import QuadrotorDynamics2D
        return QuadrotorDynamics2D()
    # elif base_type == "autorally_dynamics_carsim":
    #     from robot_planning.environment.dynamics.autorally_dynamics.autorally_dynamics_carsim import (
    #         AutoRallyDynamicsCarsim,
    #     )
    #     return AutoRallyDynamicsCarsim()
    elif base_type == "dubins_3d":
        from robot_planning.environment.dynamics.Dubins3D import Dubins3D
        return Dubins3D()
    else:
        raise ValueError("dynamics type {} not recognized".format(base_type))


def cost_evaluator_factory_base(base_type):
    if base_type == "quadratic_cost":
        from robot_planning.environment.cost_evaluators import QuadraticCostEvaluator

        return QuadraticCostEvaluator()
    elif base_type == "autorally_mppi_cost_evaluator":
        from robot_planning.environment.cost_evaluators import (
            AutorallyMPPICostEvaluator,
        )

        return AutorallyMPPICostEvaluator()
    elif base_type == "autorally_mppi_cbf_cost_evaluator":
        from robot_planning.environment.cost_evaluators import (
            AutorallyMPPICBFCostEvaluator,
        )

        return AutorallyMPPICBFCostEvaluator()
    elif base_type == "autorally_mppi_ncbf_cost_evaluator":
        from robot_planning.environment.ncbf import AutorallyMPPINCBFCostEvaluator

        return AutorallyMPPINCBFCostEvaluator()
    elif base_type == "terminal_cost":
        from robot_planning.environment.cost_evaluators import TerminalCostEvaluator

        return TerminalCostEvaluator()
    elif base_type == "abstract_cost":
        from robot_planning.environment.cost_evaluators import AbstractCostEvaluator

        return AbstractCostEvaluator()
    elif base_type == "mppi_cost_evaluator":
        from robot_planning.environment.cost_evaluators import MPPICostEvaluator

        return MPPICostEvaluator()
    elif base_type == "quadrotor_2d_cbf_cost_evaluator":
        from robot_planning.environment.cost_evaluators import Quadrotor2DCBFCostEvaluator

        return Quadrotor2DCBFCostEvaluator()
    elif base_type == "quadrotor_2d_mppi_ncbf_cost_evaluator":
        from robot_planning.environment.ncbf_quadrotor import QuadrotorMPPINCBFCostEvaluator

        return QuadrotorMPPINCBFCostEvaluator()
    elif base_type == "dubins_cbf_cost_evaluator":
        from robot_planning.environment.cost_evaluators import DubinsCBFCostEvaluator

        return DubinsCBFCostEvaluator()
    else:
        raise ValueError("cost_evaluator type {} not recognized".format(base_type))


def collision_checker_factory_base(base_type):
    if base_type == "bicycle_model_collision_checker":
        from robot_planning.environment.collision_checker import (
            BicycleModelCollisionChecker,
        )

        return BicycleModelCollisionChecker()
    elif base_type == "point_collision_checker":
        from robot_planning.environment.collision_checker import PointCollisionChecker

        return PointCollisionChecker()
    elif base_type == "autorally_collision_checker":
        from robot_planning.environment.collision_checker import (
            AutorallyCollisionChecker,
        )

        return AutorallyCollisionChecker()
    elif base_type == "mppi_collision_checker":
        from robot_planning.environment.collision_checker import MPPICollisionChecker

        return MPPICollisionChecker()
    elif base_type == "quadrotor_2d_collision_checker":
        from robot_planning.environment.collision_checker import Quadrotor2DCollisionChecker

        return Quadrotor2DCollisionChecker()
    else:
        raise ValueError("collsion_checker type {} not recognized".format(base_type))


def renderer_factory_base(base_type):
    if base_type == "MPPImatplotlib":
        from robot_planning.environment.renderers import MPPIMatplotlibRenderer

        return MPPIMatplotlibRenderer()
    elif base_type == "CSSMPCmatplotlib":
        from robot_planning.environment.renderers import CSSMPCMatplotlibRenderer

        return CSSMPCMatplotlibRenderer()
    elif base_type == "autorally_matplotlib":
        # from robot_planning.environment.renderers import AutorallyMatplotlibRenderer
        from robot_planning.environment.mpl_renderer import AutorallyMatplotlibRenderer

        return AutorallyMatplotlibRenderer()
    elif base_type == "clbf_autorally_matplotlib":
        from robot_planning.environment.renderers import CLBFAutorallyMatplotlibRenderer

        return CLBFAutorallyMatplotlibRenderer()
    elif base_type == "Envmatplotlib":
        from robot_planning.environment.renderers import EnvMatplotlibRenderer

        return EnvMatplotlibRenderer()
    elif base_type == "quadrotor_2d_matplotlib":
        from robot_planning.environment.renderers import Quadrotor2DMatplotlibRenderer

        return Quadrotor2DMatplotlibRenderer()
    elif base_type == "quadrotor_2d_matplotlib_blit":
        from robot_planning.environment.quad_renderer import QuadrotorMatplotlibRendererBlit

        return QuadrotorMatplotlibRendererBlit()
    elif base_type == "dubins_matplotlib":
        from robot_planning.environment.renderers import DubinsMatplotlibRenderer

        return DubinsMatplotlibRenderer()
    elif base_type == "dubins_matplotlib_blit":
        from robot_planning.environment.dubins_renderer import DubinsMatplotlibRendererBlit

        return DubinsMatplotlibRendererBlit()
    else:
        raise ValueError("Visualizer type {} not recognized".format(base_type))


def goal_checker_factory_base(base_type):
    if base_type == "state_space_goal_checker":
        from robot_planning.environment.goal_checker import StateSpaceGoalChecker

        return StateSpaceGoalChecker()
    elif base_type == "flex_state_space_goal_checker":
        from robot_planning.environment.goal_checker import FlexStateSpaceGoalChecker

        return FlexStateSpaceGoalChecker()
    elif base_type == "position_goal_checker":
        from robot_planning.environment.goal_checker import PositionGoalChecker

        return PositionGoalChecker()
    elif base_type == "autorally_cartesian_goal_checker":
        from robot_planning.environment.goal_checker import (
            AutorallyCartesianGoalChecker,
        )

        return AutorallyCartesianGoalChecker()
    elif base_type == "quadrotor_2d_goal_checker":
        from robot_planning.environment.goal_checker import (
            QuadrotorCartesianGoalChecker,
        )

        return QuadrotorCartesianGoalChecker()
    elif base_type == "dubins_cartesian_goal_checker":
        from robot_planning.environment.goal_checker import DubinsCartesianGoalChecker

        return DubinsCartesianGoalChecker()
    else:
        raise ValueError("goal_checker type {} not recognized".format(base_type))


def robot_factory_base(base_type):
    if base_type == "simulated_robot":
        from robot_planning.environment.robots.simulated_robot import SimulatedRobot

        return SimulatedRobot()
    if base_type == "abstract_robot":
        from robot_planning.environment.robots.abstract_robot import AbstractRobot

        return AbstractRobot()
    else:
        raise ValueError("robot type {} is not recognized".format(base_type))


def kinematics_factory_base(base_type):
    if base_type == "bicycle_model_kinematics":
        from robot_planning.environment.kinematics.simulated_kinematics import (
            BicycleModelKinematics,
        )

        return BicycleModelKinematics()
    elif base_type == "point_kinematics":
        from robot_planning.environment.kinematics.simulated_kinematics import (
            PointKinematics,
        )

        return PointKinematics()
    elif base_type == "quadrotor_kinematics_2d":
        from robot_planning.environment.kinematics.simulated_kinematics import (
            QuadrotorKinematics2D,
        )

        return QuadrotorKinematics2D()
    else:
        raise ValueError("kinematics type {} is not recognized".format(base_type))


def controller_factory_base(base_type):
    if base_type == "MPPI":
        from robot_planning.controllers.MPPI.MPPI import MPPI

        return MPPI()
    elif base_type == "CLBF_MPPI":
        from robot_planning.controllers.MPPI.CLBF_MPPI import CLBF_MPPI

        return CLBF_MPPI()
    elif base_type == "Quadrotor2D_OpenLoop":
        from robot_planning.controllers.Quadrotor2DOpenLoop import Quadrotor2DOpenLoop

        return Quadrotor2DOpenLoop()
    elif base_type == "CEMMPC":
        from robot_planning.controllers.MPPI.MPPI import CEMMPC

        return CEMMPC()

    # elif base_type == "CSSMPC":
    #     from robot_planning.controllers.CSSMPC.CSSMPC import CSSMPC
    #     return CSSMPC()
    # elif base_type == "CSSMPC_Autorally":
    #     from robot_planning.controllers.CSSMPC.CSSMPC import CSSMPCAutorally
    #     return CSSMPCAutorally()
    # elif base_type == "MPPICS_SMPC":
    #     from robot_planning.controllers.MPPICS.MPPICS_SMPC import MPPICS_SMPC
    #     return MPPICS_SMPC()
    # elif base_type == "MPPICS":
    #     from robot_planning.controllers.MPPICS.MPPICS import MPPICS
    #     return MPPICS()
    else:
        raise ValueError("controller type {} is not recognized".format(base_type))


def stochastic_trajectories_sampler_factory_base(base_type):
    if base_type == "MPPI_stochastic_trajectories_sampler":
        from robot_planning.controllers.MPPI.stochastic_trajectories_sampler import (
            MPPIStochasticTrajectoriesSampler,
        )

        return MPPIStochasticTrajectoriesSampler()
    elif base_type == "MPPI_CBF_stochastic_trajectories_sampler":
        from robot_planning.controllers.MPPI.stochastic_trajectories_sampler import (
            MPPICBFStochasticTrajectoriesSampler,
        )

        return MPPICBFStochasticTrajectoriesSampler()
    elif base_type == "MPPI_NCBF_stochastic_trajectories_sampler":
        from robot_planning.environment.ncbf import MPPINCBFStochasticTrajectoriesSampler

        return MPPINCBFStochasticTrajectoriesSampler()
    elif base_type == "MPPI_NCBF_inefficient_stochastic_trajectories_sampler":
        from robot_planning.environment.ncbf import MPPINCBFStochasticTrajectoriesSamplerInefficient

        return MPPINCBFStochasticTrajectoriesSamplerInefficient()
    elif base_type == "RAMPPI_stochastic_trajectories_sampler":
        from robot_planning.controllers.MPPI.stochastic_trajectories_sampler import (
            RAMPPIStochasticTrajectoriesSampler,
        )

        return RAMPPIStochasticTrajectoriesSampler()
    elif base_type == "CCMPPI_stochastic_trajectories_sampler_slow_loop":
        from robot_planning.controllers.MPPI.stochastic_trajectories_sampler import (
            CCMPPIStochasticTrajectoriesSamplerSLowLoop,
        )

        return CCMPPIStochasticTrajectoriesSamplerSLowLoop()
    elif base_type == "CCMPPI_stochastic_trajectories_sampler":
        from robot_planning.controllers.MPPI.stochastic_trajectories_sampler import (
            CCMPPIStochasticTrajectoriesSampler,
        )

        return CCMPPIStochasticTrajectoriesSampler()
    elif base_type == "CCMPPI_stochastic_trajectories_sampler_gpu":
        from robot_planning.controllers.MPPI.stochastic_trajectories_sampler_gpu import (
            CCMPPIStochasticTrajectoriesSamplerGpu,
        )

        return CCMPPIStochasticTrajectoriesSamplerGpu()
    elif base_type == "MPPI_stochastic_trajectories_sampler_slow_loop":
        from robot_planning.controllers.MPPI.stochastic_trajectories_sampler import (
            MPPIStochasticTrajectoriesSamplerSlowLoop,
        )

        return MPPIStochasticTrajectoriesSamplerSlowLoop()
    elif base_type == "MPPI_parallel_stochastic_trajectories_sampler_multiprocessing":
        from robot_planning.controllers.MPPI.stochastic_trajectories_sampler import (
            MPPIParallelStochasticTrajectoriesSamplerMultiprocessing,
        )

        return MPPIParallelStochasticTrajectoriesSamplerMultiprocessing()
    elif base_type == "MPPI_stochastic_trajectories_sampler_gpu":
        from robot_planning.controllers.MPPI.stochastic_trajectories_sampler_gpu import (
            MPPIStochasticTrajectoriesSamplerGpu,
        )

        return MPPIStochasticTrajectoriesSamplerGpu()
    else:
        raise ValueError(
            "stochastic_trajectories_sampler type {} is not recognized".format(
                base_type
            )
        )


def noise_sampler_factory_base(base_type):
    if base_type == "gaussian_noise_sampler":
        from robot_planning.controllers.MPPI.noise_sampler import GaussianNoiseSampler

        return GaussianNoiseSampler()
    else:
        raise ValueError("noise_sampler type {} is not recognized".format(base_type))

    if base_type == "cuda_gaussian_noise_sampler":
        from robot_planning.controllers.MPPI.noise_sampler import (
            CudaGaussianNoiseSampler,
        )

        return CudaGaussianNoiseSampler()
    else:
        raise ValueError("noise_sampler type {} is not recognized".format(base_type))


# def rl_agent_factory_base(base_type):
#     if base_type == "maddpg":
#         from robot_planning.trainers.rl_agents.maddpg_agent import MADDPG_Agent
#         return MADDPG_Agent()
#     else:
#         raise ValueError("agent type {} is not recognized".format(base_type))


# def environment_factory_base(base_type):
#     if base_type == "multi_agent_environment":
#         from robot_planning.environment.environment import MultiAgentEnvironment
#         return MultiAgentEnvironment()
#     if base_type == "abstract_environment":
#         from robot_planning.environment.environment import AbstractEnvironment
#         return AbstractEnvironment()
#     else:
#         raise ValueError("environment type {} is not recognized".format(base_type))


def observer_factory_base(base_type):
    # if base_type == "full_state_observer":
    #     from robot_planning.environment.observer import FullStateObserver
    #     return FullStateObserver()
    # if base_type == "local_state_observer":
    #     from robot_planning.environment.observer import LocalStateObserver
    #     return LocalStateObserver()
    # if base_type == "abstract_full_state_observer":
    #     from robot_planning.environment.observer import AbstractFullStateObserver
    #     return AbstractFullStateObserver()
    # else:
    raise ValueError("observer type {} is not recognized".format(base_type))


def configs_generator_factory_base(base_type):
    if base_type == "MPPI_configs_generator":
        from robot_planning.batch_experimentation.configs_generator import (
            MPPIConfigsGenerator,
        )

        return MPPIConfigsGenerator()
    elif base_type == "autorally_CSSMPC_configs_generator":
        from robot_planning.batch_experimentation.configs_generator import (
            AutorallyCSSMPCConfigsGenerator,
        )

        return AutorallyCSSMPCConfigsGenerator()
    elif base_type == "autorally_MPPI_configs_generator":
        from robot_planning.batch_experimentation.configs_generator import (
            AutorallyMPPIConfigsGenerator,
        )

        return AutorallyMPPIConfigsGenerator()
    else:
        raise ValueError("config_generator type {} is not recognized".format(base_type))


def logger_factory_base(base_type):
    if base_type == "MPPI_logger":
        from robot_planning.batch_experimentation.loggers import MPPILogger

        return MPPILogger()
    elif base_type == "Autorally_logger":
        from robot_planning.batch_experimentation.loggers import AutorallyLogger

        return AutorallyLogger()
    elif base_type == "Autorally_npz_logger":
        from robot_planning.batch_experimentation.loggers import AutorallyNpzLogger

        return AutorallyNpzLogger()
    elif base_type == "Autorally_MPPI_logger":
        from robot_planning.batch_experimentation.loggers import AutorallyMPPILogger

        return AutorallyMPPILogger()
    elif base_type == "Autorally_CSSMPC_logger":
        from robot_planning.batch_experimentation.loggers import AutorallyCSSMPCLogger

        return AutorallyCSSMPCLogger()
    elif base_type == "drone_2d_npz_logger":
        from robot_planning.batch_experimentation.loggers import Drone2DNpzLogger

        return Drone2DNpzLogger()
    elif base_type == "drone_3d_npz_logger":
        from robot_planning.batch_experimentation.loggers import Drone3DNpzLogger

        return Drone3DNpzLogger()
    else:
        raise ValueError("logger type {} is not recognized".format(base_type))


def dynamics_linearizer_factory_base(base_type):
    if base_type == "numpy_dynamics_linearizer":
        from robot_planning.environment.dynamics_linearizer import (
            NumpyDynamicsLinearizer,
        )

        return NumpyDynamicsLinearizer()
    elif base_type == "ccmppi_numpy_dynamics_linearizer":
        from robot_planning.environment.dynamics_linearizer import (
            CCMPPINumpyDynamicsLinearizer,
        )

        return CCMPPINumpyDynamicsLinearizer()
    else:
        raise ValueError(
            "dynamics_linearizer type {} is not recognized".format(base_type)
        )


def covariance_steering_helper_factory_base(base_type):
    if base_type == "cvxpy_covariance_steering_helper":
        from robot_planning.controllers.helpers.covariance_steering_helper import (
            CvxpyCovarianceSteeringHelper,
        )

        return CvxpyCovarianceSteeringHelper()
    else:
        raise ValueError(
            "covariance_steering_helper type {} is not recognized".format(base_type)
        )


def dynamic_action_bounds_generator_factory_base(base_type):
    if base_type == "dynamic_subgoal_bounds_generator":
        from robot_planning.environment.dynamic_action_bounds_generators import (
            DynamicSubgoalBoundsGenerator,
        )

        return DynamicSubgoalBoundsGenerator()
    else:
        raise ValueError(
            "dynamic_action_bounds_generator type {} is not recognized".format(
                base_type
            )
        )

def barrier_net_factory_base(base_type):
    if base_type == "barrier_nn":
        from robot_planning.environment.barriers.barrier_net import (
            BarrierNN,
        )
        return BarrierNN()