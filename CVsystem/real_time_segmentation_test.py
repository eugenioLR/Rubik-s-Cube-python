from image_to_cube import *
from video_input import *
import time
from scipy.signal.signaltools import wiener
import traceback



close_flag = False

def on_close(event):
    global close_flag
    close_flag = True

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
    im = ax.imshow(I_rgb)
    im2 = ax2.imshow(I_rgb, cmap='Greys_r')
    im3 = ax3.imshow(I_rgb)
    plt.show()

    phone = True

    a.start()
    while time.time() - start < 200 and not close_flag:

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
        I_bw = binarize(frame)
        contours = find_contours(I_bw, debug=True)

        get_ordered_colors(frame,contours, debug = True)
        frame_masked = isolate_stickers(frame)

        im.set_data(frame_masked)
        im2.set_data(binarize(frame))
        im3.set_data(frame)

        fig.canvas.draw_idle()#plt.draw()
        fig.canvas.flush_events()


        time.sleep(1/15)
except KeyboardInterrupt:
    pass
except Exception:
    traceback.print_exc()
a.stop()
