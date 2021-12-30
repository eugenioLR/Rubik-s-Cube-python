import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from image_to_cube import *


def draw_lines(theta, tol, threshold):
    #lines = cv2.HoughLines(I_border, max(angle-(angle*ang_tol),0), angle+(angle*ang_tol), 150, None, 0,0)
    lines = cv2.HoughLines(I_border, 1, np.pi/180, threshold, None, min_theta=theta-tol, max_theta=theta+tol,)
    if lines is not None:
        for i in range(len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a_t = math.cos(theta)
            b_t = math.sin(theta)
            x0 = a_t * rho
            y0 = b_t * rho
            pt1 = (int(x0 + 1000*(-b_t)), int(y0 + 1000*(a_t)))
            pt2 = (int(x0 - 1000*(-b_t)), int(y0 - 1000*(a_t)))
            plt.plot((pt1[0], pt2[0]), (pt1[1], pt2[1]))

if __name__ == '__main__':
    cube_photo_color = cv2.imread('rubiks_cube_photo.png', cv2.IMREAD_COLOR)
    cube_photo = cv2.medianBlur(cube_photo_color, 9)
    I_border = borders(cube_photo)

    #ang_tol = np.pi/15
    ang_tol=0.01

    draw_lines(0, ang_tol, 55)
    draw_lines(0.65, ang_tol, 50)
    draw_lines(np.pi/4, ang_tol, 50)


    plt.imshow(I_border, cmap='Greys_r')
    plt.show()
