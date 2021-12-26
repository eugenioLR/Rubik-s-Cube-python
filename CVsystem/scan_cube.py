import sys
sys.path.append("..")

from Cube import *
from image_to_cube import *
from video_input import *
import time
import traceback
from scipy.spatial import distance

close_flag = False

def on_close(event):
    global close_flag
    close_flag = True


# Color ranges in HSV
color_range = {
    "white":  [(0, 360), (0, 10), (90, 100)],
    "red":    [(360-340, 10), (35, 100), (70, 100)],
    "green":  [(80, 140), (50, 100), (25, 100)],
    "blue":   [(171, 260), (50, 100), (25, 100)],
    "yellow": [(50, 65), (35, 100), (40, 100)],
    "orange": [(14, 50), (70, 100), (60, 100)],
}

color_centroids_map = {
    "white":  [180, 5, 95],
    "red":    [360, 85, 85],
    "green":  [110, 75, 62],
    "blue":   [215, 75, 62],
    "yellow": [57, 95, 95],
    "orange": [32, 90, 90],
}

color_centroids = np.array([[180, 5, 95],[360, 85, 85],[110, 75, 62],[215, 75, 62],[57, 95, 95],[32, 90, 90]])



color_names = {0:"white",1:"red",2:"green",3:"blue",4:"yellow",5:"orange"}

# same as isolate_stickers(frame)?

def get_color_name(hsv_color):
    """
    Recieve a color as a 3 component vector
    The color components come in range 0-255
    """
    hue = (hsv_color[0]*360)//255
    sat = (hsv_color[1]*100)//255
    val = (hsv_color[2]*100)//255

    result = ""

    if sat < 40 and val > 80:
        result = "blanco"
    elif hue < 10 and hue > 351:
        result = "rojo"
    elif hue < 10 and hue > 40:
        result = "naranja"
    elif hue < 40 and hue > 85:
        result = "amarillo"
    elif hue < 85 and hue > 169:
        result = "verde"
    elif hue < 169 and hue > 258:
        result = "azul"
    else:
        result = "otro"

    return result

def get_ordered_colors(I_rgb):
    I_hsv = cv2.cvtColor(I_rgb, cv2.COLOR_RGB2HSV)

    I_bw = binarize(I_hsv)
    contours = find_contours(I_bw, debug=True)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [10, 10])
    for i in contours:
        I_sticker = I_hsv.copy()
        I_mask = np.zeros(I_bw.shape)
        cv2.drawContours(I_mask, contours, -1, 255, cv2.FILLED)
        I_mask = cv2.morphologyEx(I_mask, cv2.MORPH_ERODE, kernel)
        I_sticker[I_mask!=255, :] = 0
        h = I_sticker[:,:,0].sum()/np.count_nonzero(I_sticker[:,:,0])
        s = I_sticker[:,:,1].sum()/np.count_nonzero(I_sticker[:,:,1])
        v = I_sticker[:,:,2].sum()/np.count_nonzero(I_sticker[:,:,2])
        print(get_color_name(color))


# FEATURE EXTRACTION FOR AR, https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/


a = WebcamVideoStream()
try:

    plt.ion()

    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', on_close)
    ax = plt.subplot()
    I_rgb = cv2.imread("rubiks_cube_photo.png")
    im = ax.imshow(I_rgb)
    plt.show()

    phone = True

    a.start()
    while not close_flag:

        frame = a.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if phone:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        #frame[:,:,2] = cv2.equalizeHist(frame[:,:,2])
        #frame[:,:,1] = cv2.equalizeHist(frame[:,:,1])
        for i in range(3):
            frame[:,:,i] = cv2.equalizeHist(frame[:,:,i])


        #frame[:,:,2] = cv2.GaussianBlur(frame[:,:,2], (5,5), 0)
        #frame[:,:,1] = cv2.GaussianBlur(frame[:,:,1], (5,5), 0)
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10,10,7,21)

        #frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)


        #kernel = np.ones((4, 4), np.uint8)
        #dilation = cv2.dilate(gray, kernel, iterations=1)

        #blur = cv2.GaussianBlur(dilation, (5, 5), 0)

        #frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

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
