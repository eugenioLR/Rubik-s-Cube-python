# import the necessary packages
from threading import Thread
import cv2
import time
from datetime import datetime

class WebcamVideoStream:
	def __init__(self, fps=30, src=0):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
		self.fps = fps

	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			self.grabbed, self.frame = self.stream.read(); time.sleep(1/self.fps)


	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

if __name__ == '__main__':
    a = WebcamVideoStream()

    frame = cv2.rotate(a.read(), cv2.ROTATE_90_CLOCKWISE)

    date_string = time.strftime("%Y-%m-%d_%H-%M-%S")

    cv2.imwrite(f"cube_samples/cube_photo{date_string}.png", frame)
