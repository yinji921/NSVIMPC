# Safe Beyond the Horizon: Efficient Sampling-based MPC with Neural Control Barrier Functions

This repository contains the code for reproducing the simulations in the paper:
**"Safe Beyond the Horizon: Efficient Sampling-based MPC with Neural Control Barrier Functions"**
## Repository Structure

### Core Simulation Scripts
- **AutoRally experiments**:
  - `robot_planning/scripts/run_AutoRally_CBF_MPPI_for_experiment.py` - Main AutoRally simulation
  - `robot_planning/scripts/run_ablations.py` - Ablation studies
  
- **Drone/Quadrotor2D experiments**:
  - `robot_planning/scripts/run_quadrotor2d.py` - 2D quadrotor simulation
  
- **Additional experiments**:
  - `robot_planning/scripts/run_dubins3d.py` - Dubins car simulation

### Neural CBF Training
- `ncbf/scripts/train_offline.py` - Train NCBF for AutoRally
- `ncbf/scripts/train_offline_drone.py` - Train NCBF for drone
- `ncbf/scripts/eval_ncbf.py` - Evaluate trained NCBF

### Data Collection (in functional_scripts/)
- `robot_planning/scripts/functional_scripts/collect_offline_dset.py` - Collect AutoRally data
- `robot_planning/scripts/functional_scripts/collect_offline_dset_drone.py` - Collect drone data

### Utility Scripts
- `robot_planning/scripts/functional_scripts/` - Data processing and analysis utilities
- `ncbf/scripts/functional_scripts/` - Visualization and debugging tools

## Installation

### Recommended Method: Conda Environment

The easiest way to install all dependencies with compatible versions is to use the provided conda environment:

```bash
# Create the conda environment with all dependencies
conda env create -f environment.yml

# Activate the environment
conda activate nsvimpc

# Install the local og package
pip install -e ./og
```

This will install all dependencies including:
- JAX 0.4.24 with CUDA support for GPU-accelerated MPPI controllers
- TensorFlow Probability 0.25.0 for distributions (JAX substrate)  
- NumPy 1.26.4 (compatible with JAX 0.4.24)
- PyTorch 2.5.1, matplotlib, scipy, and other scientific computing packages
- All other dependencies with tested, compatible versions

### Alternative: Manual Installation

If you prefer manual installation, you can use pip with the requirements file:

```bash
# Install core dependencies (may have version conflicts)
pip install -r robot_planning/requirements.txt

# Install the local og package
pip install -e ./og
```

**Note**: Manual pip installation is not recommended due to potential dependency conflicts between JAX, TensorFlow Probability, and other packages. The conda environment approach provides tested, compatible versions. You are also welcome to install the most updated version of these packages with the updated og package, which can be found: https://github.com/oswinso/og

## Quick Start

### Running AutoRally Simulation
```bash
cd robot_planning
python scripts/run_AutoRally_CBF_MPPI_for_experiment.py
```

### Running Drone Simulation
```bash
cd robot_planning
python scripts/run_quadrotor2d.py
```

### Training Neural CBF
The NCBF training process consists of multiple steps:

1. **Collect data**:
```bash
cd robot_planning/scripts
python collect_offline_dset.py
```
This collects trajectories and saves the dataset to `data/raw_data.pkl`.

2. **Process data**:
```bash
# From the root directory
python robot_planning/scripts/functional_scripts/raw_to_dset.py data/raw_data.pkl
```
This converts `data/raw_data.pkl` to `data/dset.pkl`.

3. **Train offline**:
```bash
# From the root directory
python ncbf/scripts/train_offline.py
```
This trains the model and saves checkpoints in the `runs` folder.

4. **Use the trained NCBF**: Update the checkpoint path in your scripts to point to the trained model.

The MPPI controller has been tuned to give satisfactory performance.

For the sake of testing the algorithms, there are several parameters of interests that you can play with such that the MPPI becomes more or less reliable. 
In the file tests/configs/test_run_Autorally_MPPI.cfg, generally:
1. Increasing or decreasing the parameter "number_of_trajectories" under [my_stochastic_trajectories_sampler1] will make the controller more and less stable, respectively.
2. The diagonal terms of "covariance" under [my_noise_sampler1] determines the degree of exploration. Increasing them means searching in a larger area with sparser trajectory sampling, given a fixed value of "number_of_trajectories". For instance, making the diagonal terms of "covariance" to be 0.01 will result in extremely small exploration and causes the vehicle to crash the boundaries.
3. The first dimension of "goal_state" under [my_goal_checker_for_tracking] is the target speed that MPPI tracks, please feel free to play with it.
4. "trajectories_rendering" under [renderer1] is a switch for turning on and off rendering of sample trajectories. Turn it on to observe the difference that tuning "number_of_trajectories" or "covariance" makes. In our testing scripts, we sometimes use a boolean parameter "render" to overide the rendering switch.

## Citation

If you use this code in your research, please cite our papers:

```bibtex
@inproceedings{yin2025safe,
  title        = {Safe Beyond the Horizon: Efficient Sampling-based MPC with Neural Control Barrier Functions},
  author       = {Ji Yin and Oswin So and Eric Yang Yu and Chuchu Fan and Panagiotis Tsiotras},
  booktitle    = {Proceedings of Robotics: Science and Systems (RSS)},
  year         = {2025},
  address      = {Los Angeles, CA},
  month        = {June},
  day          = {21--25}
}

@ARTICLE{yin2023shield,
  author={Yin, Ji and Dawson, Charles and Fan, Chuchu and Tsiotras, Panagiotis},
  journal={IEEE Robotics and Automation Letters}, 
  title={Shield Model Predictive Path Integral: A Computationally Efficient Robust MPC Method Using Control Barrier Functions}, 
  year={2023},
  volume={8},
  number={11},
  pages={7106-7113},
  keywords={Trajectory;Safety;Heuristic algorithms;Costs;Planning;Vehicle dynamics;Uncertainty;Autonomous driving;computational efficiency;optimal control and motion planning;vehicle safety},
  doi={10.1109/LRA.2023.3315211}}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

