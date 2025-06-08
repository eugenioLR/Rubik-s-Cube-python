import argparse
from .cv_system.cube_calibration import Cube_calibrator
# from .cv_system.real_time_segmentation_test as seg_test
from .solvers.IDA_NNSolver.Rubik_heuristic_NN import *
import json

def launch_solver(config_path):
    # Reading the config file
    json_config = open(config_path)
    config = json.loads(json_config.read())
    json_config.close()

    # The main program
    cubeCal = Cube_calibrator(config)
    cubeCal.main()

    # Segmentation tests for debugging
    #seg_test.main()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/config_nn.json")
    args = parser.parse_args()

    launch_solver(args.config)

if __name__ == '__main__':
    main()