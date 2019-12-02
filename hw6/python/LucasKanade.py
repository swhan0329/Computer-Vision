import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   p: movement vector dx, dy
    
    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)
    x1,y1,x2,y2 = rect

    # put your implementation here
    dp=1
    iter = 0

    Y = np.arange(It.shape[0])
    X = np.arange(It.shape[1])

    c = np.linspace(x1, x2)
    r = np.linspace(y1, y2)
    cc, rr = np.meshgrid(c, r)

    interp = RectBivariateSpline(Y, X, It)
    T = interp.ev(rr, cc)

    Y1 = np.arange(It1.shape[0])
    X1 = np.arange(It1.shape[1])

    interp1 = RectBivariateSpline(Y1, X1, It1)

    while np.sqrt(np.square(dp).sum()) > threshold and iter < maxIters:
        iter += 1

        # warp image
        dx, dy = p[0], p[1]
        x1_w, y1_w, x2_w, y2_w = x1 + dx, y1 + dy, x2 + dx, y2 + dy

        cw = np.linspace(x1_w, x2_w)
        rw = np.linspace(y1_w, y2_w)
        ccw, rrw = np.meshgrid(cw, rw)

        I_x = interp1.ev(rrw, ccw, dy=1)
        I_y = interp1.ev(rrw, ccw, dx=1)

        warpImg = interp1.ev(rrw, ccw)

        b = T - warpImg
        b = b.reshape(-1, 1)

        A = np.vstack((I_x.ravel(), I_y.ravel())).T

        H = A.T @ A

        dp = np.linalg.inv(H) @ (A.T) @ b

        p[0] += dp[0, 0]
        p[1] += dp[1, 0]
        #print("p:",p)
    #print(iter)
    return p

