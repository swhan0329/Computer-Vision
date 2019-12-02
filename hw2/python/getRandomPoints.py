import numpy as np
import cv2 as cv

def get_random_points(I, alpha):

    # -----fill in your implementation here --------
    h = np.size(I,0)
    w = np.size(I,1)

    points = np.zeros([alpha,2], dtype=int)
    hV = np.random.randint(0,h,alpha)
    wV = np.random.randint(0,w,alpha)
    points[:,0] = wV
    points[:,1] = hV

    # ----------------------------------------------
    return points

if __name__ == "__main__":
    img = cv.imread('../data/airport/sun_aerinlrdodkqnypz.jpg')
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    alpha = 50
    #print(get_random_points(img,alpha))
    points = get_random_points(img,alpha)
    red = [0, 0, 255]
    for point in points:
        cv.circle(img, tuple(point), 1, red, -1)
    cv.imshow("%d Random Points" % (alpha), img)
    cv.waitKey(0)