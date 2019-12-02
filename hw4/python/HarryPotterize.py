import numpy as np
import cv2
import skimage.io 
import skimage.color
#Import necessary functions
from python.planarH import *
from python.matchPics import *


#Write script for Q2.2.4
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

matches, locs1, locs2 = matchPics(cv_cover,cv_desk,1)

new_locs1 = np.zeros((np.size(matches,axis=0),2), dtype=np.float32)
new_locs2 = np.zeros((np.size(matches,axis=0),2), dtype=np.float32)

for i in range(0,np.size(matches,axis=0)):
    new_locs1[i,:] = locs1[matches[i,0],:]
    new_locs2[i, :] = locs2[matches[i, 1], :]

bestH2to1, inliers = computeH_ransac(new_locs2,new_locs1)
h, status = cv2.findHomography(new_locs1, new_locs2, cv2.RANSAC)

hp_cover = cv2.resize(hp_cover, dsize=(np.shape(cv_cover)[1], np.shape(cv_cover)[0]), interpolation=cv2.INTER_AREA)
composit_img = compositeH(h,hp_cover,cv_desk)

cv2.imshow("composit_img",composit_img)
cv2.waitKey(0)
