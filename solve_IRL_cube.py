from CVsystem.cube_calibration import Cube_calibrator
import CVsystem.real_time_segmentation_test as seg_test
from IDA_NNSolver.Rubik_heuristic_NN import *
import json

if __name__ == '__main__':
    json_config = open("config.json")
    config = json.loads(json_config.read())
    json_config.close()

    # The main program
    cubeCal = Cube_calibrator(config)
    cubeCal.main()

    # Segmentation tests for debugging
    #seg_test.main()
