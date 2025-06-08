import cv2
import time
from datetime import datetime
from threading import Thread

class WebcamVideoStream:
    def __init__(self, fps=30, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        
        self.stopped = False
        self.fps = fps

    def start(self):
        Thread(target=self.update, args=()).start()

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read();
            time.sleep(1/self.fps)


    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

if __name__ == '__main__':
    a = WebcamVideoStream()

    frame = cv2.rotate(a.read(), cv2.ROTATE_90_CLOCKWISE)

    date_string = time.strftime("%Y-%m-%d_%H-%M-%S")

    cv2.imwrite(f"cube_samples/cube_photo{date_string}.png", frame)
