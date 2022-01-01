# Sticker size = 1.5 cm
from . import outlier_detection as outlier
from . import *
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

    # Add black bars to the the image to ensure the fill is correct
    I_filled_large = np.vstack([np.zeros([I.shape[1]]), I_filled])
    I_filled_large = np.hstack([np.zeros([I.shape[0]+1,1]), I_filled_large]).astype(np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(I_filled_large, mask, (0,0), 255)

    # Restore the image to it's original size
    I_filled = I_filled_large[1:,1:]

    # Invert floodfilled image
    I_filled_inv = cv2.bitwise_not(I_filled)

    # Combine the two images to get the foreground.
    return I | I_filled_inv

def binarize(I_hsv):
    I_value = cv2.normalize(I_hsv[:,:,2], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    #th, I_bw = cv2.threshold(I_value, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    I_bw = cv2.adaptiveThreshold(I_value, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [10, 10])
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, [6, 6])

    I_bw = cv2.morphologyEx(I_bw, cv2.MORPH_OPEN, kernel1)
    I_bw = cv2.morphologyEx(I_bw, cv2.MORPH_ERODE, kernel2)

    return I_bw

def borders(I_hsv):
    I_value = cv2.normalize(I_hsv[:,:,2], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)


    I_border = cv2.Canny(I_value, 100, 200)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [4, 4])
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [1, 1])

    I_inv = np.invert(I_border)
    I_inv = cv2.morphologyEx(I_inv, cv2.MORPH_OPEN, kernel1)
    I_border = np.invert(I_inv)

    #I_border = cv2.morphologyEx(I_inv, cv2.MORPH_DILATE, kernel2)

    return I_border

def hough_lines(I_border):
    lines = cv2.HoughLines(I_border, 1, np.pi/180, 150, None, 0,0)


def find_contours(I_bw, debug = False):
    im_h, im_w = I_bw.shape

    # We find the contours
    contours, hierarchy = cv2.findContours(I_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    boundary_mask = np.zeros(len(contours)) == 1
    properties = np.zeros([7, len(contours)])
    for i in range(len(contours)):
        # We obtain the bounding Rectangle
        x, y, w, h = cv2.boundingRect(contours[i])
        (_, _), (w_min, h_min), theta = cv2.minAreaRect(contours[i])

        h = max(h, 0.000000001)
        h_min = max(h_min, 0.000000001)
        h_min = max(h_min, 0.000000001)

        aspect_ratio = w_min/h_min
        area = cv2.contourArea(contours[i])
        hull = cv2.convexHull(contours[i], False)
        convex_area = cv2.contourArea(hull)
        convex_area = max(convex_area, 0.000000001)
        perimeter = cv2.arcLength(contours[i], True)
        perimeter = max(perimeter, 0.000000001)
        approx_poly = cv2.approxPolyDP(contours[i], 0.15*perimeter, True)

        rectangularity = 16 * (area/perimeter**2)
        #rectangularity = w*h/max(area, 0.000001)

        #properties[0, i] = len(approx_poly)
        properties[0, i] = rectangularity           # Rectangularity (modification of circularity)
        properties[1, i] = aspect_ratio             # aspect ratio
        properties[2, i] = area                     # area
        properties[3, i] = area/convex_area         # convexity index
        properties[4, i] = perimeter                # perimeter
        properties[5, i] = x-w/2                    # coordinates in the image
        properties[6, i] = y-h/2

    # Restrict to rectangles
    boundary_mask = (properties[0, :] > 0.85) & (properties[0, :] < 1.175)

    # The aspect ratio can't be too big
    boundary_mask = boundary_mask & (properties[1, :] < 5) & (properties[1, :] > 1/5)

    # The area can't be too big or too small relative to the image size
    boundary_mask = boundary_mask & (properties[2, :] > 0.0025*im_h*im_w)
    boundary_mask = boundary_mask & (properties[2, :] < 0.03*im_h*im_w)

    # The boundary must be convex
    boundary_mask = boundary_mask & (properties[3, :] > 0.875)

    if np.count_nonzero(boundary_mask) > 9:
        # Detect outliers based on their position, area and convexity
        # -position: the cube's stickers must be close to each other
        # -area: the cube's stickers have roughly the same area
        # -convexity: all the cube's stickers are convex

        prop_aux = properties[[3, 2, 5, 6]][:, boundary_mask]

        distance_filter = outlier.relative_density(prop_aux, 9, 0.75, 'euclidean')

        if debug:
            print(f"{np.count_nonzero(distance_filter)} boundaries were outliers")

        boundary_mask[boundary_mask] = ~distance_filter

    if debug:
        print(f"{np.count_nonzero(boundary_mask)} of the {len(contours)} boundaries were conserved")

    return [contours[i] for i in range(len(boundary_mask)) if boundary_mask[i]], properties[[5, 6]][:, boundary_mask]

def sticker_mask(I_hsv, method="bin", debug = False):
    if method == "bin":
        I_bw = binarize(I_hsv)
    elif method == "border":
        I_border = borders(I_hsv)
        I_bw = imfill(I_border)

    contours, positions = find_contours(I_bw, debug=debug)

    I_result = np.zeros(I_bw.shape)

    cv2.drawContours(I_result, contours, -1, 255, cv2.FILLED)

    return I_result==255


#def kmeans_colors(I_masked):

def isolate_stickers(I_hsv, method="bin"):
    mask = sticker_mask(I_hsv, method).astype(np.uint8)

    mask = mask == 0

    I_result = I_hsv.copy()

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

def get_ordered_colors(I_hsv, contours, debug = False):
    face = np.zeros(1000)
    positions = -np.ones([2, 9])
    face_positions = None
    avg_w = 0
    avg_h = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, [4, 4])
    for i in range(len(contours)):
        I_sticker = I_hsv.copy()
        I_mask = np.zeros([I_hsv.shape[0], I_hsv.shape[1]], np.uint8)
        cv2.drawContours(I_mask, [contours[i]], -1, 255, cv2.FILLED)
        I_mask = cv2.morphologyEx(I_mask, cv2.MORPH_ERODE, kernel)

        I_sticker[np.isnan(I_sticker)] = 0
        I_sticker = I_sticker[I_mask==255, :]

        #color = I_sticker.mean(axis=0)
        color = np.median(I_sticker, axis=0)

        x, y, w, h = cv2.boundingRect(contours[i])

        avg_w += w
        avg_h += h

        if i < 9:
            positions[:, i] = np.array([x,y]).T
            face[i] = get_color_name(color)


        if debug:
            # display color
            h, s, v = color
            r, g, b = colorsys.hsv_to_rgb(((h*360)//255)/255,s/255,v/255)
            r, g, b = (np.array([r, g, b])*255).astype(np.uint8)

            print(f"color {i}: {color_names[get_color_name(color)]} = {(h*360)//176},{(s*100)//255},{(v*100)//255}, \033[48;2;{r};{g};{b}m:)\033[48;2;0;0;0m")
            #print(f"color {i}: {get_color_name(color)} = {h},{s},{v}, \033[48;2;{r};{g};{b}m:)\033[48;2;0;0;0m")

    if len(contours) == 9:
        avg_w /= 9
        avg_h /= 9

        face_order_y = np.argsort(positions[1, :])
        positions_grid = np.reshape(positions[:,face_order_y], [2,3,3])
        face = np.reshape(face[face_order_y], [3,3])
        face_order_x = np.argsort(positions_grid[0,:,:], axis=1)
        face = np.take_along_axis(face, face_order_x, axis=1)

        face_positions = positions[:,face_order_y] + np.vstack([avg_w, avg_h])/2
        face_positions[:, :3] = face_positions[:, face_order_x[0,:]]
        face_positions[:,3:6] = face_positions[:, face_order_x[1,:]+3]
        face_positions[:,6: ] = face_positions[:, face_order_x[2,:]+6]

        if debug:
            print(face)
    else:
        face = None

    return np.array(face), face_positions
