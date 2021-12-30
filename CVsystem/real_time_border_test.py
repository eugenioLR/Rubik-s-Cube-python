from image_to_cube import *
from video_input import *
import time
from scipy.signal.signaltools import wiener
import traceback
import math



close_flag = False

def on_close(event):
    global close_flag
    close_flag = True

def draw_lines(theta, tol, threshold, subplt):
    #lines = cv2.HoughLines(I_border, 1, np.pi/180, threshold, None, min_theta=theta-tol, max_theta=theta+tol,)
    lines = cv2.HoughLinesP(I_border, 1, np.pi/180, threshold, 50, 10)
    if lines is not None:
        for i in range(len(lines)):
            line = lines[i][0]
            subplt.plot((line[0], line[2]), (line[1], line[3]))

a = WebcamVideoStream()
try:
    start = time.time()

    plt.ion()

    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', on_close)
    ax = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)

    I_rgb = cv2.imread("rubiks_cube_photo.png")
    im = ax.imshow(I_rgb, cmap='Greys_r')
    line_plot, = ax.plot([0,0],[0,0], linewidth=3)
    im2 = ax2.imshow(I_rgb, cmap='Greys_r')
    im3 = ax3.imshow(I_rgb)
    plt.show()

    phone = True

    a.start()
    while time.time() - start < 200 and not close_flag:

        ax.clear()

        frame = a.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



        if phone:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        #frame[:,:,2] = cv2.equalizeHist(frame[:,:,2])
        #frame[:,:,1] = cv2.equalizeHist(frame[:,:,1])
        #for i in range(3):
        #    frame[:,:,i] = cv2.equalizeHist(frame[:,:,i])


        #frame[:,:,2] = cv2.GaussianBlur(frame[:,:,2], (5,5), 0)
        #frame[:,:,1] = cv2.GaussianBlur(frame[:,:,1], (5,5), 0)

        #frame = cv2.fastNlMeansDenoisingColored(frame, None, 10,10,7,21)
        #frame[:,:,1] = cv2.boxFilter(frame[:,:,1], 0, [7,7], None, [-1, -1], False, cv2.BORDER_DEFAULT)
        frame = cv2.medianBlur(frame, 9)

        #frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)


        #kernel = np.ones((4, 4), np.uint8)
        #dilation = cv2.dilate(gray, kernel, iterations=1)

        #blur = cv2.GaussianBlur(dilation, (5, 5), 0)

        #frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

        #get_ordered_colors(frame, debug = False)
        I_border = borders(frame)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [6, 6])
        I_border = cv2.morphologyEx(I_border, cv2.MORPH_CLOSE, kernel_close)
        I_filled = imfill(I_border)

        kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [6, 6])
        I_filled = cv2.morphologyEx(I_filled, cv2.MORPH_ERODE, kernel_erosion)

        #ang_tol=0.01

        #draw_lines(0, ang_tol, 55, ax)
        #draw_lines(np.pi*3/4, ang_tol, 50, ax)
        #draw_lines(np.pi*5/4, ang_tol, 100, ax)


        im = ax.imshow(I_filled, cmap="Greys_r")
        #im.set_data(I_border)
        im2.set_data(I_border)
        im3.set_data(frame)

        fig.canvas.draw_idle()#plt.draw()
        fig.canvas.flush_events()



        time.sleep(1/15)
        ax.clear()
except KeyboardInterrupt:
    pass
except Exception:
    traceback.print_exc()
a.stop()
