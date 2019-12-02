"""
Homework 5
Submission Functions
"""

# import packages here
from scipy.signal import convolve2d
import numpy as np
import helper as hlp
import cv2
from scipy.spatial.distance import cdist
np.set_printoptions(precision=4, suppress=True)
"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    # replace pass by your implementation
    N = np.size(pts1, axis=0)

    MT = np.diag([1 / M, 1 / M, 1])

    pts1_ = pts1 / M
    pts2_ = pts2 / M

    A = np.zeros((N, 9))
    for i in range(N):
        x = pts1_[i, 0]
        #x = 479 - x
        y = pts1_[i, 1]
        xp = pts2_[i, 0]
        #xp = 479 - xp
        yp = pts2_[i, 1]
        A[i, :] = [x * xp, x * yp, x, y * xp, y * yp, y, xp, yp, 1]

    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape((3,3))

    U, S, V = np.linalg.svd(F)
    S[-1] = 0

    F = U @ np.diag(S) @ V

    F = MT.T @ F @ MT

    F = hlp.refineF(F, pts1, pts2)
    #print("F:",F)
    return F


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    # replace pass by your implementation
    #print(pts1.shape)
    if pts1.shape[0] == 2:
        x1,y1 = pts1[0], pts1[1]
    else:
        x1, y1 = pts1[0,0], pts1[0,1]

    if im1.ndim == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if im2.ndim == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    h, w = im1.shape

    lp = F@[x1,y1,1]
    lp = lp/lp[2]
    s = np.sqrt(lp[0]*lp[0]+lp[1]*lp[1])
    lp = lp/s

    im1 = np.double(im1)
    im2 = np.double(im2)

    thr = 5

    minx = max(0,x1-thr)
    maxx = min(w,x1+thr)
    miny = max(0,y1-thr)
    maxy = min(h,y1+thr)
    #print(minx, maxx, miny,maxy)

    window = 5
    target = im1[y1-window:y1+window,x1-window:x1+window]
    target = target.flatten()
    target = target.reshape(-1, 1)
    minV = -1
    minpx = 0
    minpy = 0

    for i in range(minx,maxx):
        #print("x:",i)
        x = i
        if lp[1] == 0:
            y = y1
        else:
            y = (-lp[2]-lp[0]*x)/lp[1]

        x = np.floor(x)
        y = np.floor(y)
        x = x.astype(int)
        y = y.astype(int)
        if (x-window<1 or y-window<1 or x+window>w or y+window>h or (x-x1)**2+(y-y1)**2 > 1000):
            continue
        new = im2[y-window:y+window,x-window:x+window]
        new = new.flatten()
        new = new.reshape(-1,1)

        d = cdist(target.T, new.T, 'euclidean')
        if minV == -1 or d < minV:
            minV = d
            minpx = x
            minpy = y

    for i in range(miny,maxy):
        #print("y:",i)
        y = i
        if lp[1] == 0:
            x = x1
        else:
            x = (-lp[2]-lp[0]*y)/lp[1]

        x = np.floor(x)
        y = np.floor(y)
        x = x.astype(int)
        y = y.astype(int)
        if (x-window<1 or y-window<1 or x+window>w or y+window>h or (x-x1)**2+(y-y1)**2 > 1000):
            continue
        new = im2[y-window:y+window,x-window:x+window]
        new = new.flatten()
        new = new.reshape(-1, 1)

        d = cdist(target.T, new.T, 'euclidean')
        if (minV == -1 or d <minV):
            minV = d
            minpx = x
            minpy = y

    x2 = minpx
    y2 = minpy

    ptr2 = np.zeros((1,2))
    ptr2[0,0] = x2
    ptr2[0,1] = y2

    return ptr2


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # replace pass by your implementation
    E = K2.T@F@K1

    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    # replace pass by your implementation
    N = np.size(pts1, axis=0)

    P1_0 = P1[0,:]
    P1_1 = P1[1,:]
    P1_2 = P1[2,:]
    P2_0 = P2[0,:]
    P2_1 = P2[1,:]
    P2_2 = P2[2,:]

    pp = np.zeros((N, 4))
    A = np.zeros((4, 4))
    pts3d = np.zeros((N, 3))
    for i in range(N):
        x, y = pts1[i,0], pts1[i,1]
        xp,yp = pts2[i,0], pts2[i,1]

        A[0,:] = y * P1_2 - P1_1
        A[1,:] = P1_0 - x * P1_2
        A[2, :] = yp * P2_2 - P2_1
        A[3, :] = P2_0 -xp * P2_2

        U, S, V = np.linalg.svd(A.T@A)
        p = V.T[:,-1]
        pts3d[i,:]=p[0:3]/p[3]
        pp[i, :] = p / p[3]

    return pts3d, pp

"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    c1=-(np.linalg.inv(np.dot(K1,R1)))@(np.dot(K1,t1))
    c2=-(np.linalg.inv(np.dot(K2,R2)))@(np.dot(K2,t2))

    r1=(c1-c2)/np.linalg.norm(c1-c2,2)
    r2=np.cross(R1[2,:].T,r1)
    r3=np.cross(r1,r2)

    Rn = [r1, r2, r3]
    Rn= np.asarray(Rn)

    R1p = Rn
    R2p = Rn

    K1p = K2
    K2p = K2

    t1p = -Rn@c1
    t2p = -Rn@c2

    M1 = (K1p@R1p)@(np.linalg.inv(K1p@R1p))
    M2 = (K2p@R2p)@(np.linalg.inv(K2p@R2p))

    return M1,M2,K1p,K2p,R1p,R2p,t1p,t2p

"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    y,x=im1.shape
    dispM=np.zeros((y,x))
    w = (win_size-1)/2
    w = int(w)

    temp= np.zeros((max_disp))
    for yy in range(y):
        for xx in range(x):
            for d in range(max_disp):
                for i in range(-w,w):
                    for j in range(-w, w):
                        temp[d] = (im1[yy+i,xx+j]-im2[yy+i,xx+j-d])**2

            dispM[yy,xx]=np.argmin(temp)

    return dispM


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    c1=-(np.linalg.inv(np.dot(K1,R1)))@(np.dot(K1,t1))
    c2=-(np.linalg.inv(np.dot(K2,R2)))@(np.dot(K2,t2))
    y,x = dispM.shape
    depthM = np.zeros((y, x))
    b = np.linalg.norm(c1-c2,2)
    f = K1[0,0]

    for i in range(y):
        for j in range(x):
            if dispM[i,j] == 0:
                pass
            else:
                depthM[i,j] = b*f/dispM[i,j]

    return depthM

"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
