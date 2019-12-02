import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Load the two temple images and the points from data/some_corresp.npz
#I1 = io.imread('../data/im1.png')
#I2 = io.imread('../data/im2.png')
im1= io.imread('../data/im1.png')[:,:,:3]
im2= io.imread('../data/im2.png')[:,:,:3]
height, width, channels = im1.shape

some_data = np.load("../data/some_corresp.npz")
some_pts1 = some_data["pts1"]
some_pts2 = some_data["pts2"]

some_data.close()

# 2. Run eight_point to compute F
M = max(height,width)

F = sub.eight_point(some_pts1,some_pts2,M)
hlp.displayEpipolarF(im1,im2,F)

# 3. Load points in image 1 from data/temple_coords.npz
temple_data = np.load("../data/temple_coords.npz")
temple_pts1 = temple_data["pts1"]

temple_data.close()

# 4. Run epipolar_correspondences to get points in image 2
def displayEpipolarMatches(I1, I2, pts1, pts2):
    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Image 1 Coordinates')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Image 2 Coordinates')
    ax2.set_axis_off()

    ax1.plot(pts1[:,0], pts1[:,1], 'r*', MarkerSize=6)
    ax2.plot(pts2[:,0], pts2[:,1], 'r*', MarkerSize=6)

    plt.show()

N = np.size(temple_pts1,axis=0)
cor_pts1 = np.zeros((N,2))
for i in range(N):
    cor_pts1[i] = sub.epipolar_correspondences(im1, im2, F, temple_pts1[i])

#displayEpipolarMatches(im1, im1, temple_pts1, cor_pts1)
#hlp.epipolarMatchGUI(im1, im1, F)

# 5. Compute the camera projection matrix P1
data = np.load("../data/intrinsics.npz")
K1 = data["K1"]
K2 = data["K2"]

data.close()

I = np.diag([1,1, 1])
P = np.zeros((3,4))
P[:,:-1]=I

P1 = np.matmul(K1, P)

# 6. Use camera2 to get 4 camera projection matrices P2
E = sub.essential_matrix(F, K1, K2)
RT = hlp.camera2(E)

P2_num=RT.shape[2]

# 7. Run triangulate using the projection matrices
min_err = np.inf
temp =0
temp_idx = 0
for i in range(P2_num):
    P2_c=np.matmul(K2,RT[:,:,i])
    pts3d,_ = sub.triangulate(P1, temple_pts1, P2_c, cor_pts1)

    # 8. Figure out the correct P2
    num = sum(pts3d[:,2]>=0)
    if temp < num:
        temp = num
        temp_idx = i

print("what a number of P2:",temp_idx)
P2=np.matmul(K2,RT[:,:,temp_idx])
pts3d, _ = sub.triangulate(P1, temple_pts1, P2, cor_pts1)

P2=np.matmul(K2,RT[:,:,temp_idx])
pts3d_some, pp_some = sub.triangulate(P1, some_pts1, P2, some_pts2)

pts1_proj = P1 @ pp_some.T
pts1_proj /= pts1_proj[-1,:]
pts2_proj = P2 @ pp_some.T
pts2_proj /= pts2_proj[-1,:]

def reproj_error(ori,proj):
    err = (proj - ori) * (proj - ori)
    return err

err1 = np.sum(reproj_error(some_pts1, pts1_proj[0:2,:].T))/N
err2 = np.sum(reproj_error(some_pts2, pts2_proj[0:2,:].T))/N

total_err = err1 + err2
print("err1:",err1)
print("err2:",err2)
print("Total_err:",total_err)

# 9. Scatter plot the correct 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
x_min, x_max = np.min(pts3d[:,0]), np.max(pts3d[:,0])
y_min, y_max = np.min(pts3d[:,1]), np.max(pts3d[:,1])
z_min, z_max = np.min(pts3d[:,2]), np.max(pts3d[:,2])

ax.set_xlim3d(x_min, x_max)
ax.set_ylim3d(y_min, y_max)
ax.set_zlim3d(z_min, z_max)
ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c='b', marker='o')
plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
np.savez('../data/extrinsics.npz', R1=P[:,0:-1], R2=RT[:,:,temp_idx][:,0:-1], t1=P[:,-1], t2=RT[:,:,temp_idx][:,-1])