import sys
sys.path.append("..")

from Cube import *

from image_to_cube import *
from video_input import *
import time
from scipy.signal.signaltools import wienerW
import traceback

class Cube_calibrator:
    def __init__(self):
        self.current_face = -1
        self.calibrated = False
        self.cube = Cube(3)
        self.cam = WebcamVideoStream()

    def calibrate_cube(self, fps, phone=True):
        self.cam.start()
        frame_time = 1/fps
        start = time.time()
        while time.time() - start < 5:
            # Measure time for FPS control
            frame_start = time.time()



            # Image Aquisition
            frame = self.cam.read()

            if phone:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Pre-processing
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            #frame = cv2.fastNlMeansDenoisingColored(frame, None, 10,10,7,21) # Too slow for real time, but gives the best results
            frame = cv2.medianBlur(frame, 9)

            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)


            # Extraction of characteristics
            frame_bw = binarize(frame)
            contours = find_contours(frame_bw, debug=True)


            # Description
            get_ordered_colors(frame, contours, debug = True)
            frame_masked = isolate_stickers(frame)



            # Measure time for FPS control
            frame_end = time.time()

            # Limit FPS
            time_passed = frame_end - frame_start
            if time_passed < frame_time:
                time.sleep(frame_time-time_passed)

        self.cam.stop()
        self.current_face = -1
        self.calibrated = True
