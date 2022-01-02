import traceback
from .video_input import WebcamVideoStream
from .image_processing_utils import *
from . import *
import time

close_flag = False

def on_close(event):
    global close_flag
    close_flag = True
def main():
    a = WebcamVideoStream()
    try:
        start = time.time()

        plt.ion()

        fig = plt.figure()
        fig.canvas.mpl_connect('close_event', on_close)
        ax = fig.add_gridspec(2,3)
        ax1 = plt.subplot(2,3,1)
        ax2 = plt.subplot(2,3,2)
        ax3 = plt.subplot(2,3,3)
        #ax4 = plt.subplot(2,3,4)
        ax5 = plt.subplot(2,3,5)
        ax6 = plt.subplot(2,3,6)
        plt.subplots_adjust(bottom=0.05, right=0.9, top=0.95)

        path = str(Path(__file__).resolve().parent) + "/"
        I_rgb = cv2.imread(path + "rubiks_cube_photo.png")
        im1 = ax1.imshow(I_rgb)
        im2 = ax2.imshow(I_rgb, cmap='Greys_r')
        im3 = ax3.imshow(I_rgb)
        #im4 = ax4.imshow(I_rgb)
        im5 = ax5.imshow(I_rgb, cmap='Greys_r')
        im6 = ax6.imshow(I_rgb)
        plt.show()

        phone = True

        a.start()
        while time.time() - start < 200 and not close_flag:
            ## Image Aquisition
            frame_original = a.read()

            # The original image will be in BGR, we transform it to RGB
            frame_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB)

            if phone:
                frame_original = cv2.rotate(frame_original, cv2.ROTATE_90_CLOCKWISE)


            ## Pre-processing
            frame_hsv = cv2.cvtColor(frame_original, cv2.COLOR_RGB2HSV)

            #frame = cv2.fastNlMeansDenoisingColored(frame, None, 10,10,7,21) # Too slow for real time, but gives the best results
            frame_hsv = cv2.medianBlur(frame_hsv, 9)
            #frame_hsv[:,:,2] = cv2.normalize(frame_hsv[:,:,2], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

            #increase contrast
            frame_hsv[:,:,2] = cv2.convertScaleAbs(frame_hsv[:,:,2], alpha=1.2, beta=0.8)

            #frame_hsv[:,:,2] = cv2.equalizeHist(frame_hsv[:,:,2])
            #frame[:,:,1] = cv2.equalizeHist(frame[:,:,1])
            #for i in range(3):
            #    frame[:,:,i] = cv2.equalizeHist(frame[:,:,i])


            #frame[:,:,2] = cv2.GaussianBlur(frame[:,:,2], (5,5), 0)
            #frame[:,:,1] = cv2.GaussianBlur(frame[:,:,1], (5,5), 0)

            #frame = cv2.fastNlMeansDenoisingColored(frame, None, 10,10,7,21)
            #frame[:,:,1] = cv2.boxFilter(frame[:,:,1], 0, [7,7], None, [-1, -1], False, cv2.BORDER_DEFAULT)

            frame_preprocessed = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2RGB)

            # USING ADAPTATIVE THRESHOLD
            frame_bw = binarize(frame_hsv)
            contours, __ = find_contours(frame_bw, debug=False)

            get_ordered_colors(frame_hsv, contours, debug=False)
            frame_masked_hsv = isolate_stickers(frame_hsv, "bin")
            frame_masked = cv2.cvtColor(frame_masked_hsv, cv2.COLOR_HSV2RGB)

            # USING CANNY BORDERS
            frame_border = filled_borders(frame_hsv)

            contours_b, __ = find_contours(frame_border, debug=True)

            get_ordered_colors(frame_hsv, contours, debug=False)
            frame_masked_hsv_b = isolate_stickers(frame_hsv, "border")
            frame_masked_b = cv2.cvtColor(frame_masked_hsv_b, cv2.COLOR_HSV2RGB)

            plt.subplot(2,3,1)
            im1.set_data(frame_preprocessed)
            plt.title("image input")

            plt.subplot(2,3,2)
            im2.set_data(frame_bw)
            plt.title("binarized image")

            plt.subplot(2,3,3)
            im3.set_data(frame_masked)
            plt.title("masked with binarize")

            plt.subplot(2,3,5)
            im5.set_data(frame_border)
            plt.title("borders filled")

            plt.subplot(2,3,6)
            im6.set_data(frame_masked_b)
            plt.title("masked with borders")

            fig.canvas.draw_idle()#plt.draw()
            fig.canvas.flush_events()


            time.sleep(1/15)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    a.stop()
