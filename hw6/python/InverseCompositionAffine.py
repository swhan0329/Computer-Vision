import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros((6,1))
    x1,y1,x2,y2 = rect

    # put your implementation here
    dp = 1

    x = np.arange(0, It.shape[0])
    y = np.arange(0, It.shape[1])

    c = np.linspace(x1, x2)
    r = np.linspace(y1, y2)
    cc, rr = np.meshgrid(c, r)

    # I here is the gradient of the template
    interp = RectBivariateSpline(x, y, It)
    T = interp.ev(rr, cc)

    I_x = interp.ev(rr, cc, dy=1)
    I_y = interp.ev(rr, cc, dx=1)

    A = np.vstack((I_x.ravel() * cc.ravel(), I_x.ravel() * rr.ravel(), I_x.ravel(), I_y.ravel() * cc.ravel(),
                   I_y.ravel() * rr.ravel(), I_y.ravel())).T
    H = A.T @ A

    iter = 0

    Y1 = np.arange(It1.shape[0])
    X1 = np.arange(It1.shape[1])

    interp1 = RectBivariateSpline(Y1, X1, It1)

    while np.sqrt(np.square(dp).sum()) > threshold and iter < maxIters:
        iter += 1

        M = np.array([[1.0 + p[0, 0], p[1, 0], p[2, 0]],
                      [p[3, 0], 1.0 + p[4, 0], p[5, 0]]])

        x1_w = M[0, 0] * x1 + M[0, 1] * y1 + M[0, 2]
        y1_w = M[1, 0] * x1 + M[1, 1] * y1 + M[1, 2]
        x2_w = M[0, 0] * x2 + M[0, 1] * y2 + M[0, 2]
        y2_w = M[1, 0] * x2 + M[1, 1] * y2 + M[1, 2]

        cw = np.linspace(x1_w, x2_w)
        rw = np.linspace(y1_w, y2_w)
        ccw, rrw = np.meshgrid(cw, rw)

        warpImg = interp1.ev(rrw, ccw)

        b = T - warpImg
        b = b.reshape(-1, 1)

        dp = np.linalg.inv(H) @ (A.T) @ b

        # update parameters
        p[0] += dp[0, 0]
        p[1] += dp[1, 0]
        p[2] += dp[2, 0]
        p[3] += dp[3, 0]
        p[4] += dp[4, 0]
        p[5] += dp[5, 0]

    print(iter)
    # reshape the output affine matrix
    M = np.array([[1.0+p[0], p[1],    p[2]],
                 [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)

    return M
