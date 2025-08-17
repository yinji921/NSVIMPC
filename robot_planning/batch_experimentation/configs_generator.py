import configparser as ConfigParser
import numpy as np
import ast
import copy


class ConfigsGenerator:
    def __init__(
        self,
        config_data=None,
        batch_experiments_sizes=None,
        parameter_ranges=None,
        template_config=None,
    ):
        self.config_data = config_data
        self.batch_experiments_sizes = batch_experiments_sizes
        self.parameter_ranges = parameter_ranges
        self.template_config = template_config

    def initialize_from_config(self, config_data, section_name):
        if config_data.has_option(section_name, "batch_experiments_sizes"):
            self.batch_experiments_sizes = np.asarray(
                ast.literal_eval(
                    config_data.get(section_name, "batch_experiments_sizes")
                ),
                dtype=int,
            )
        if config_data.has_option(section_name, "parameter_ranges"):
            self.parameter_ranges = np.asarray(
                ast.literal_eval(config_data.get(section_name, "parameter_ranges")),
                dtype=np.float64,
            )
        if config_data.has_option(section_name, "template_config_path"):
            template_config_path = config_data.get(section_name, "template_config_path")
            self.template_config = ConfigParser.ConfigParser()
            self.template_config.read(template_config_path)

    def generate(self):
        raise NotImplementedError

    def generate_parameters_sets(
        self, parameters_values, parameters_sets, layer_idx, para_set
    ):
        # This function utilizes DFS to find all possible parameter values combinations and return the set
        if layer_idx > len(parameters_values) - 1:
            return
        for i in range(len(parameters_values[layer_idx])):
            para_set.append(parameters_values[layer_idx][i])
            if layer_idx == len(parameters_values) - 1:
                parameters_sets.append(copy.deepcopy(para_set))
            else:
                self.generate_parameters_sets(
                    parameters_values, parameters_sets, layer_idx + 1, para_set
                )
            para_set.pop()
        return parameters_sets

    def create_run_batch_config_file(
        self, parameters_sets, experiment_names, run_batch_section_name
    ):
        run_batch_config = ConfigParser.ConfigParser()
        run_batch_config[run_batch_section_name] = {}
        run_batch_config[run_batch_section_name]["experiment_names"] = "["
        for experiment_name in experiment_names:
            if run_batch_config[run_batch_section_name]["experiment_names"] != "[":
                run_batch_config[run_batch_section_name]["experiment_names"] += ", "
            run_batch_config[run_batch_section_name]["experiment_names"] += (
                "'" + experiment_name + "'"
            )
        run_batch_config[run_batch_section_name]["experiment_names"] += "]"
        run_batch_config_name = run_batch_section_name
        with open("configs/" + run_batch_config_name + ".cfg", "w") as configfile:
            run_batch_config.write(configfile)
            configfile.close()


class MPPIConfigsGenerator(ConfigsGenerator):
    def __init__(self, config_data=None):
        ConfigsGenerator.__init__(self, config_data)

    def initialize_from_config(self, config_data, section_name):
        ConfigsGenerator.initialize_from_config(self, config_data, section_name)

    def generate_managing_config(self):
        run_batch_MPPI_new_config = ConfigParser.ConfigParser()
        run_batch_MPPI_new_config["run_batch_MPPI"] = {}

    def generate_batch_experiment_configs(self, template_config=None):
        if template_config is not None:
            self.template_config = template_config
        if self.template_config is None:
            raise ValueError("Missing a valid template configuration file!")
        parameters_values = []
        for i in range(len(self.parameter_ranges)):
            ith_parameter_range = self.parameter_ranges[i]
            ith_parameter_values = np.linspace(
                ith_parameter_range[0],
                ith_parameter_range[1],
                num=self.batch_experiments_sizes[i],
            )
            parameters_values.append(ith_parameter_values)
        parameters_sets = self.generate_parameters_sets(parameters_values, [], 0, [])
        experiment_names = []
        for set in parameters_sets:
            # Note: set[0], set[1], and set[2] are x, y and radius of the first obstacle,
            # set[3], set[4] and set[5] are x, y and radius of the second obstacle
            # set[6] is the inverse temperature, set[7] is the uncontrolled_trajectories_portion,
            # and set[8] is the number_of_trajectories of the stochastic_trajectories_sampler
            new_config = copy.copy(self.template_config)
            obstacles = [[set[0], set[1]], [set[3], set[4]]]
            obstacle_radius = [set[2], set[5]]
            inverse_temperature = set[6]
            uncontrolled_trajectories_portion = set[7]
            number_of_trajectories = set[8]

            new_config["my_collision_checker1"]["obstacles"] = str(obstacles)
            new_config["my_collision_checker1"]["obstacles_radius"] = str(
                obstacle_radius
            )
            new_config["my_controller1"]["inverse_temperature"] = str(
                inverse_temperature
            )
            new_config["my_stochastic_trajectories_sampler1"][
                "uncontrolled_trajectories_portion"
            ] = str(uncontrolled_trajectories_portion)
            new_config["my_stochastic_trajectories_sampler1"][
                "number_of_trajectories"
            ] = str(number_of_trajectories)
            experiment_name = "exp_obs1x{}_obs1y{}_obs1r{}_obs2x{}_obs2y{}_obs2r{}_gamma{}_alpha{}_Numtraj{}_MPPI".format(
                set[0], set[1], set[2], set[3], set[4], set[5], set[6], set[7], set[8]
            )  # TODO: you can extend this to adjust more parameters
            new_config["logger"]["experiment_name"] = str(experiment_name)
            experiment_config_name = experiment_name + ".cfg"
            with open(
                "configs/batch_configs/" + experiment_config_name, "w"
            ) as configfile:
                new_config.write(configfile)
            experiment_names.append(experiment_name)

        self.create_run_batch_config_file(
            parameters_sets, experiment_names, "run_batch_MPPI"
        )


class AutorallyCSSMPCConfigsGenerator(ConfigsGenerator):
    def __init__(self, config_data=None):
        ConfigsGenerator.__init__(self, config_data)

    def initialize_from_config(self, config_data, section_name):
        ConfigsGenerator.initialize_from_config(self, config_data, section_name)
        if config_data.has_option(section_name, "dynamics_tuned"):
            self.dynamics_tuned = config_data.get(section_name, "dynamics_tuned")
        else:
            self.dynamics_tuned = "sim_dynamics1"

    def generate_managing_config(self):
        run_batch_MPPI_new_config = ConfigParser.ConfigParser()
        run_batch_MPPI_new_config["run_batch_Autorally_CSSMPC"] = {}

    def generate_batch_experiment_configs(self, template_config=None):
        if template_config is not None:
            self.template_config = template_config
        if self.template_config is None:
            raise ValueError("Missing a valid template configuration file!")
        parameters_values = []
        if self.parameter_ranges is None:
            tire_B = self.template_config.get("sim_dynamics1", "tire_B")
            tire_C = self.template_config.get("sim_dynamics1", "tire_C")
            tire_D = self.template_config.get("sim_dynamics1", "tire_D")
            kSteering = self.template_config.get("sim_dynamics1", "kSteering")
            cSteering = self.template_config.get("sim_dynamics1", "cSteering")
            throttle_factor = self.template_config.get(
                "sim_dynamics1", "throttle_factor"
            )
            mean_parameter_set = np.array(
                (tire_B, tire_C, tire_D, kSteering, cSteering, throttle_factor),
                dtype=float,
            )
            parameters_sets = []
            for ii in range(100):
                disturbances = 0.1 * np.random.randn(6)
                parameters_set = mean_parameter_set + disturbances * mean_parameter_set
                parameters_sets.append(list(parameters_set))
        else:
            for i in range(len(self.parameter_ranges)):
                ith_parameter_range = self.parameter_ranges[i]
                ith_parameter_values = np.linspace(
                    ith_parameter_range[0],
                    ith_parameter_range[1],
                    num=self.batch_experiments_sizes[i],
                )
                parameters_values.append(ith_parameter_values)
            parameters_sets = self.generate_parameters_sets(
                parameters_values, [], 0, []
            )
        experiment_names = []
        for set in parameters_sets:
            # Note: set[0] is tire_B value, set[1] is tire_C value, set[2] is tire_D value, set[3] is kSteering value,
            # set[4] is cSteering value, and set[5] is throttle_factor value
            new_config = copy.copy(self.template_config)
            tire_B = set[0]
            tire_C = set[1]
            tire_D = set[2]
            kSteering = set[3]
            cSteering = set[4]
            throttle_factor = set[5]

            new_config[self.dynamics_tuned]["tire_B"] = str(tire_B)
            new_config[self.dynamics_tuned]["tire_C"] = str(tire_C)
            new_config[self.dynamics_tuned]["tire_D"] = str(tire_D)
            new_config[self.dynamics_tuned]["kSteering"] = str(kSteering)
            new_config[self.dynamics_tuned]["cSteering"] = str(cSteering)
            new_config[self.dynamics_tuned]["throttle_factor"] = str(throttle_factor)
            experiment_name = (
                "exp_tB{}_tC{}_tD{}_kS{}_cS{}_tf{}_Autorally_CSSMPC".format(
                    tire_B, tire_C, tire_D, kSteering, cSteering, throttle_factor
                )
            )  # TODO: you can extend this to adjust more parameters
            new_config["logger"]["experiment_name"] = str(experiment_name)
            experiment_config_name = experiment_name + ".cfg"
            with open(
                "configs/batch_configs/" + experiment_config_name, "w"
            ) as configfile:
                new_config.write(configfile)
            experiment_names.append(experiment_name)

        self.create_run_batch_config_file(
            parameters_sets, experiment_names, "run_batch_Autorally_CSSMPC"
        )


class AutorallyMPPIConfigsGenerator(ConfigsGenerator):
    def __init__(self, config_data=None):
        ConfigsGenerator.__init__(self, config_data)

    def initialize_from_config(self, config_data, section_name):
        ConfigsGenerator.initialize_from_config(self, config_data, section_name)
        if config_data.has_option(section_name, "dynamics_tuned"):
            self.dynamics_tuned = config_data.get(section_name, "dynamics_tuned")
        else:
            self.dynamics_tuned = "sim_dynamics1"

    def generate_batch_experiment_configs(self, template_config=None):
        if template_config is not None:
            self.template_config = template_config
        if self.template_config is None:
            raise ValueError("Missing a valid template configuration file!")
        parameters_values = []
        if self.parameter_ranges is None:
            tire_B = self.template_config.get("sim_dynamics1", "tire_B")
            tire_C = self.template_config.get("sim_dynamics1", "tire_C")
            tire_D = self.template_config.get("sim_dynamics1", "tire_D")
            kSteering = self.template_config.get("sim_dynamics1", "kSteering")
            cSteering = self.template_config.get("sim_dynamics1", "cSteering")
            throttle_factor = self.template_config.get(
                "sim_dynamics1", "throttle_factor"
            )
            mean_parameter_set = np.array(
                (tire_B, tire_C, tire_D, kSteering, cSteering, throttle_factor),
                dtype=float,
            )
            parameters_sets = []
            for ii in range(100):
                disturbances = 0.1 * np.random.randn(6)
                parameters_set = mean_parameter_set + disturbances * mean_parameter_set
                parameters_sets.append(list(parameters_set))
        else:
            for i in range(len(self.parameter_ranges)):
                ith_parameter_range = self.parameter_ranges[i]
                ith_parameter_values = np.linspace(
                    ith_parameter_range[0],
                    ith_parameter_range[1],
                    num=self.batch_experiments_sizes[i],
                )
                parameters_values.append(ith_parameter_values)
            parameters_sets = self.generate_parameters_sets(
                parameters_values, [], 0, []
            )
        experiment_names = []
        for set in parameters_sets:
            # Note: set[0] is tire_B value, set[1] is tire_C value, set[2] is tire_D value, set[3] is kSteering value,
            # set[4] is cSteering value, and set[5] is throttle_factor value
            new_config = copy.copy(self.template_config)
            tire_B = set[0]
            tire_C = set[1]
            tire_D = set[2]
            kSteering = set[3]
            cSteering = set[4]
            throttle_factor = set[5]

            new_config[self.dynamics_tuned]["tire_B"] = str(tire_B)
            new_config[self.dynamics_tuned]["tire_C"] = str(tire_C)
            new_config[self.dynamics_tuned]["tire_D"] = str(tire_D)
            new_config[self.dynamics_tuned]["kSteering"] = str(kSteering)
            new_config[self.dynamics_tuned]["cSteering"] = str(cSteering)
            new_config[self.dynamics_tuned]["throttle_factor"] = str(throttle_factor)
            experiment_name = "exp_tB{}_tC{}_tD{}_kS{}_cS{}_tf{}_Autorally_MPPI".format(
                tire_B, tire_C, tire_D, kSteering, cSteering, throttle_factor
            )  # TODO: you can extend this to adjust more parameters
            new_config["logger"]["experiment_name"] = str(experiment_name)
            experiment_config_name = experiment_name + ".cfg"
            with open(
                "configs/batch_configs/" + experiment_config_name, "w"
            ) as configfile:
                new_config.write(configfile)
            experiment_names.append(experiment_name)

        self.create_run_batch_config_file(
            parameters_sets, experiment_names, "run_batch_Autorally_MPPI"
        )
