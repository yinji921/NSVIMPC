import numpy as np
import os
import matplotlib.pyplot as plt
from robot_planning.helper.utils import AUTORALLY_DYNAMICS_DIR


class BatchDataParser:
    def __init__(self):
        self.states = []
        self.crashes = []

    def parse_datas(self):
        batch_code = "0"
        directory = "/home/ji/SSD/DCSL/Autorally_MPPI_CLBF/robot_planning/experiments/Autorally_experiments/" + batch_code + "/"
        for run_file in os.listdir(directory):
            run_data = np.load(directory + run_file)
            self.states.append(run_data["states"])
            self.crashes.append(run_data["crash"])

    def print_crashes(self):
        num_crashes = sum(self.crashes)
        num_runs = len(self.crashes)
        print(
            num_crashes,
            " crashes out of ",
            num_runs,
            " runs = ",
            num_crashes / num_runs,
        )

    def print_lap_stats(self):
        num_runs = len(self.states)
        average_speeds = []
        lap_times = []
        max_speeds = []
        dt = 0.1
        for ii, states in enumerate(self.states):
            average_speed = np.mean(states[0, :])
            average_speeds.append(average_speed)
            max_speed = np.max(states[0, :])
            if self.crashes[ii] == 1:
                continue
            max_speeds.append(max_speed)
            lap_time = len(states[-1, :]) * dt
            lap_times.append(lap_time)
        print("max speed: ", np.max(max_speeds))
        print("average speed: ", np.mean(average_speeds))
        print("average lap time: ", np.mean(lap_times))

    def plot_trajectories(self):
        plt.figure()
        map_file = "/home/ji/SSD/DCSL/Autorally_MPPI_CLBF/robot_planning/environment/dynamics/autorally_dynamics/CCRF_2021-01-10.npz"
        map = np.load(map_file)
        plt.plot(map["X_in"], map["Y_in"], "k")
        plt.plot(map["X_out"], map["Y_out"], "k")
        for states in self.states:
            # plt.scatter(
            #     states[6, :], states[7, :], c=states[0, :], marker=".", vmin=1, vmax=7
            # )
            plt.plot(
                states[6, :], states[7, :], color = 'c', alpha=0.2
            )
        # cbar = plt.colorbar()
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        # cbar.set_label("speed (m/s)")
        plt.show()


if __name__ == "__main__":
    data_parser = BatchDataParser()
    data_parser.parse_datas()
    data_parser.print_crashes()
    data_parser.print_lap_stats()
    data_parser.plot_trajectories()
