# Sticker size = 1.5 cm
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import outlier_detection as outlier
from scipy.spatial import distance
import colorsys
import math
from scipy import stats
import random

sticker_size = 1.5

def imfill(I):

    # Copy the thresholded image.
    I_filled = I.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = I.shape[:2]
    mask = np.zeros((h+3, w+3), np.uint8)

    I_filled_large = np.vstack([np.zeros([I.shape[1]]), I_filled])
    I_filled_large = np.hstack([np.zeros([I.shape[0]+1,1]), I_filled_large]).astype(np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(I_filled_large, mask, (0,0), 255)

    I_filled = I_filled_large[1:,1:]

    # Invert floodfilled image
    I_filled_inv = cv2.bitwise_not(I_filled)

    # Combine the two images to get the foreground.
    return I | I_filled_inv

def binarize(I_rgb):
    I_hsv = cv2.cvtColor(I_rgb, cv2.COLOR_RGB2HSV)

    I_value = cv2.normalize(I_hsv[:,:,2], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    #I_gray = cv2.Canny(I_gray, 100, 200)
    #th, I_bw = cv2.threshold(I_value, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    I_bw = cv2.adaptiveThreshold(I_value, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, [6, 6])
    I_bw = cv2.morphologyEx(I_bw, cv2.MORPH_ERODE, kernel_erosion)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [10, 10])
    I_bw = cv2.morphologyEx(I_bw, cv2.MORPH_OPEN, kernel_close)

    return I_bw

def borders(I_rgb):
    I_hsv = cv2.cvtColor(I_rgb, cv2.COLOR_RGB2HSV)

    I_gray = cv2.normalize(I_hsv[:,:,2], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    I_border = cv2.Canny(I_gray, 100, 200)
    #I_border = cv2.Sobel(I_gray, cv2.CV_8UC1, 1, 0,ksize=5)
    #th, I_bw = cv2.threshold(I_value, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #I_bw = cv2.adaptiveThreshold(I_value, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

    #kernel_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, [6, 6])
    #I_bw = cv2.morphologyEx(I_bw, cv2.MORPH_ERODE, kernel_erosion)

    I_inv = np.invert(I_border)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [6, 6])
    I_bw = cv2.morphologyEx(I_inv, cv2.MORPH_OPEN, kernel_close)

    I_border = np.invert(I_inv)

    return I_border

def hough_lines(I_border):
    lines = cv2.HoughLines(I_border, 1, np.pi/180, 150, None, 0,0)


def find_contours(I_bw, debug = False):
    im_h, im_w = I_bw.shape
    contours, hierarchy = cv2.findContours(I_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # maybe removing outliers

    boundary_mask = np.zeros(len(contours)) == 1
    properties = np.zeros([7, len(contours)])
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        (_, _), (w_min, h_min), theta = cv2.minAreaRect(contours[i])

        h = max(h, 0.000000001)

        aspect_ratio = w_min/h_min
        area = cv2.contourArea(contours[i])
        hull = cv2.convexHull(contours[i], False)
        convex_area = cv2.contourArea(hull)
        convex_area = max(convex_area, 0.000000001)
        perimeter = cv2.arcLength(contours[i], True)
        approx_poly = cv2.approxPolyDP(contours[i], 0.15*perimeter, True)

        squarity = 16 * (area/perimeter**2)
        #squarity = area/w*h

        #properties[0, i] = len(approx_poly)
        properties[0, i] = squarity                 # Rectangularity (modification of circularity)
        properties[1, i] = aspect_ratio             # aspect ratio
        properties[2, i] = area                     # area
        properties[3, i] = area/convex_area         # convexity index
        properties[4, i] = perimeter                # perimeter
        properties[5, i] = x                        # coordinates in the image
        properties[6, i] = y

    # Restrict to paralelograms
    #boundary_mask = properties[0, :] == 4
    boundary_mask = (properties[0, :] > 0.85) & (properties[0, :] < 1.175)


    # The aspect ratio can't be too big
    boundary_mask = boundary_mask & (properties[1, :] < 5) & (properties[1, :] > 1/5)

    # The area can't be too big or too small
    boundary_mask = boundary_mask & (properties[2, :] > 0.0025*im_h*im_w)
    boundary_mask = boundary_mask & (properties[2, :] < 0.03*im_h*im_w)

    # The boundary must be convex
    boundary_mask = boundary_mask & (properties[3, :] > 0.875)

    if np.count_nonzero(boundary_mask) > 9:
        # Detect outliers based on their position, area and convexity
        # -position: the cube's stickers must be close to each other
        # -area: the cube's stickers have roughly the same area
        # -convexity: all the cube's stickers are convex
        # -squarity is not a good measuere, we put strict limits on the squarity already

        prop_aux = properties[[3, 2, 5, 6]][:, boundary_mask]

        distance_filter = outlier.relative_density(prop_aux, 9, 0.75, 'euclidean')

        if debug:
            print(f"{np.count_nonzero(distance_filter)} boundaries were outliers")

        boundary_mask[boundary_mask] = ~distance_filter

    if debug:
        print(f"{np.count_nonzero(boundary_mask)} of the {len(contours)} boundaries were conserved")

    return [contours[i] for i in range(len(boundary_mask)) if boundary_mask[i]], properties[[5, 6]][:, boundary_mask]

def sticker_mask(I_rgb, debug = False):
    I_bw = binarize(I_rgb)

    contours, positions = find_contours(I_bw, debug=debug)

    I_result = np.zeros(I_bw.shape)

    cv2.drawContours(I_result, contours, -1, 255, cv2.FILLED)

    return I_result==255


#def kmeans_colors(I_masked):

def isolate_stickers(I_rgb):
    mask = sticker_mask(I_rgb).astype(np.uint8)

    #mask = imfill(mask) == 0
    mask = mask == 0

    I_result = I_rgb.copy()

    I_result[mask, :] = 1

    return I_result


color_names = {0:"white", 1:"red", 2:"blue", 3:"orange", 4:"green", 5:"yellow", -2: "gray", -1:"purple?"}

def get_color_name(hsv_color):
    """
    Recieve a color as a 3 component vector
    The color components come in range 0-255 except for the hue which is from 0 to 179 (for some reason...)
    """
    hue = (hsv_color[0]*360)//179
    sat = (hsv_color[1]*100)//255
    val = (hsv_color[2]*100)//255

    result = -1

    if sat < 50 and val > 50:
        # White
        result = 0
    elif  sat < 50 and val < 50:
        # Gray (not used)
        result = -2
    elif hue <= 16 or hue >= 351:
        # Red
        result = 1
    elif hue > 16 and hue <= 40:
        # Orange
        result = 3
    elif hue > 40 and hue <= 85:
        # Yellow
        result = 5
    elif hue > 85 and hue <= 169:
        # Green
        result = 4
    elif hue > 151 and hue <= 260:
        # Blue
        result = 2
    else:
        result = -1

    return result

#Same as isolate_stickers(frame)?

def get_ordered_colors(I_rgb, contours, debug = False):
    I_hsv = cv2.cvtColor(I_rgb, cv2.COLOR_RGB2HSV)

    #I_bw = binarize(I_rgb)

    #contours = find_contours(I_bw, debug=True)

    face = np.zeros(9)
    positions = -np.ones([2, 9])
    face_positions = None

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, [4, 4])
    for i in range(len(contours)):
        I_sticker = I_hsv.copy()
        I_mask = np.zeros([I_rgb.shape[0], I_rgb.shape[1]], np.uint8)
        cv2.drawContours(I_mask, [contours[i]], -1, 255, cv2.FILLED)
        I_mask = cv2.morphologyEx(I_mask, cv2.MORPH_ERODE, kernel)

        I_sticker[np.isnan(I_sticker)] = 0
        I_sticker = I_sticker[I_mask==255, :]

        #color = I_sticker.mean(axis=0)
        color = np.median(I_sticker, axis=0)

        x, y, _, _ = cv2.boundingRect(contours[i])

        if i < 9:
            positions[:, i] = np.array([x,y]).T
            face[i] = get_color_name(color)


        if debug:# display color
            h, s, v = color
            r, g, b = colorsys.hsv_to_rgb(((h*360)//255)/255,s/255,v/255)
            r, g, b = (np.array([r, g, b])*255).astype(np.uint8)


            print(f"color {i}: {color_names[get_color_name(color)]} = {(h*360)//176},{(s*100)//255},{(v*100)//255}, \033[48;2;{r};{g};{b}m:)\033[48;2;0;0;0m")
            #print(f"color {i}: {get_color_name(color)} = {h},{s},{v}, \033[48;2;{r};{g};{b}m:)\033[48;2;0;0;0m")

    if len(contours) == 9:
        face_order_y = np.argsort(positions[1, :])
        positions_grid = np.reshape(positions[:,face_order_y], [2,3,3])
        face = np.reshape(face[face_order_y], [3,3])
        face_order_x = np.argsort(positions_grid[0,:,:], axis=1)
        face = np.take_along_axis(face, face_order_x, axis=1)

        face_positions = positions[:,face_order_y]
        face_positions[:, :3] = face_positions[:, face_order_x[0,:]]
        face_positions[:,3:6] = face_positions[:, face_order_x[1,:]+3]
        face_positions[:,6: ] = face_positions[:, face_order_x[2,:]+6]

        if debug:
            print(face)
    else:
        face = None

    return np.array(face), face_positions


def test_isolate_stickers():
    I_rgb = cv2.imread("rubiks_cube_photo.png")

    I_rgb = isolate_stickers(I_rgb)

    I_rgb = cv2.cvtColor(I_rgb, cv2.COLOR_BGR2HSV)

    plt.subplot(1, 3, 1)
    plt.imshow(I_rgb[:,:,0], cmap = 'hsv')
    plt.subplot(1, 3, 2)
    plt.imshow(I_rgb[:,:,1], cmap = 'Greys_r')
    plt.subplot(1, 3, 3)
    plt.imshow(I_rgb[:,:,2], cmap = 'Greys_r')
    plt.show()

def test_find_contours():
    I_rgb = cv2.imread("rubiks_cube_photo.png")

    contours = find_contours(binarize(I_rgb))

    plt.imshow(I_rgb)
    for i in range(len(contours)):
        plt.plot(contours[i][:,:,0], contours[i][:,:,1], 'w')

    plt.show()

if __name__ == '__main__':
    I_rgb = cv2.imread("rubiks_cube_photo.png")
    I_rgb = cv2.cvtColor(I_rgb, cv2.COLOR_BGR2RGB)

    I_bw = binarize(I_rgb)

    edges = cv2.Canny(I_bw, 100, 200,3)

    I_rgb_masked = isolate_stickers(I_rgb)

    plt.subplot(1,3,1)
    plt.imshow(I_rgb)
    plt.subplot(1,3,2)
    plt.imshow(I_rgb_masked)
    plt.subplot(1,3,3)
    plt.imshow(I_bw - edges, cmap='Greys_r')
    plt.show()
