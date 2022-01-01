from CVsystem.cube_calibration import Cube_calibrator
import CVsystem.real_time_segmentation_test as seg_test
from IDA_NNSolver.Rubik_heuristic_NN import *

if __name__ == '__main__':
    cubeCal = Cube_calibrator(60, 2.5)
    cubeCal.calibrate_cube()

    #seg_test.main()
