import numpy as np
import cv2 as cv
from scipy import ndimage
from utils import imfilter


def get_harris_points(I, alpha, k):

    if len(I.shape) == 3 and I.shape[2] == 3:
        I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
    if I.max() > 1.0:
        I = I / 255.0

    # -----fill in your implementation here --------
    h = np.size(I,0)
    w = np.size(I,1)

    dy, dx = np.gradient(I)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2

    # compute Harris feature strength, avoiding divide by zero
    #R = (Ixx * Iyy - Ixy**2) / (Ixx + Iyy + 1e-8)
    R = (Ixx * Iyy) -k* (Ixx + Iyy)**2

    # exclude points near the image border
    R[:16, :] = 0
    R[-16:, :] = 0
    R[:, :16] = 0
    R[:, -16:] = 0

    # non-maximum suppression in 3x3 regions
    maxH = ndimage.filters.maximum_filter(R, (5, 5))
    imgH = R * (R == maxH)

    # sort points by strength and find their positions
    sortIdx = np.argsort(imgH.flatten())[::-1]
    sortIdx = sortIdx[:alpha]
    yy = sortIdx / w
    xx = sortIdx % w
    yy = list(map(int, yy))
    xx = list(map(int, xx))

    # concatenate positions and values
    points = np.vstack((xx, yy)).transpose()

    # ----------------------------------------------
    
    return points

if __name__ == "__main__":
    img = cv.imread('../data/airport/sun_aerinlrdodkqnypz.jpg')
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    alpha = 50
    k = 0.004
    #print(get_harris_points(img,alpha,k))
    points = get_harris_points(img,alpha,k)
    red = [0, 0, 255]
    for point in points:
        cv.circle(img, tuple(point), 1, red, -1)
    cv.imshow("%d Harris Points" % (alpha), img)
    cv.waitKey(0)