import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_ROOT_DIR = ROOT_DIR + "/experiments"
DATA_ROOT_DIR = ROOT_DIR + "/data"
AUTORALLY_DYNAMICS_DIR = ROOT_DIR + "/environment/dynamics/autorally_dynamics"
SCRIPTS_DIR = ROOT_DIR + "/scripts"

if __name__ == "__main__":
    a = 0
    if a is None or a == 0:
        print("yeah")
    else:
        print("no")
