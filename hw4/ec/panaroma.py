import numpy as np
import cv2
#Import necessary functions
from python.planarH import *
from python.matchPics import *
from python.loadVid import *
from matplotlib import pyplot as plt

#Write script for Q4.2x
pano_left = cv2.imread('../data/pano_left.jpg')
pano_right = cv2.imread('../data/pano_right.jpg')

matches, locs1, locs2 = matchPics(pano_right,pano_left,1)

new_locs1 = np.zeros((np.size(matches,axis=0),2), dtype=np.float32)
new_locs2 = np.zeros((np.size(matches,axis=0),2), dtype=np.float32)

for i in range(0,np.size(matches,axis=0)):
    new_locs1[i,:] = locs1[matches[i,0],:]
    new_locs2[i, :] = locs2[matches[i, 1], :]

H, inliers = computeH_ransac(new_locs2,new_locs1)

dst = cv2.warpPerspective(pano_right,H,(pano_left.shape[1] + pano_right.shape[1], pano_left.shape[0]))
plt.subplot(122),plt.imshow(dst),plt.title("Warped Image")
plt.show()
plt.figure()
dst[0:pano_left.shape[0], 0:pano_left.shape[1]] = pano_left
cv2.imwrite("output.jpg",dst)
plt.imshow(dst)
plt.axis('off')
plt.show()