import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


if __name__ == '__main__':
    #face_reference = cv2.imread('cube_face_sample.png', cv2.IMREAD_GRAYSCALE)
    face_reference = cv2.imread('cube_face_test.png')
    face_reference = cv2.cvtColor(face_reference, cv2.COLOR_RGB2HSV)
    #face_reference = cv2.equalizeHist(face_reference[:,:,0])

    cube_photo_color = cv2.imread('rubiks_cube_photo.png')
    cube_photo = cv2.cvtColor(cube_photo_color, cv2.COLOR_RGB2HSV)
    #cube_photo = cv2.equalizeHist(cube_photo_hsv[:,:,0])

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(face_reference, None)
    kp2, des2 = orb.detectAndCompute(cube_photo, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key = lambda x:x.distance)

    I_match = cv2.drawMatches(face_reference, kp1, cube_photo, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    I_match = cv2.cvtColor(I_match, cv2.COLOR_HSV2BGR)
    plt.imshow(I_match)
    plt.show()
