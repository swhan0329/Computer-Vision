import numpy as np
import cv2
from python.matchPics import matchPics
from scipy.ndimage import rotate
from python.helper import plotMatches
import matplotlib.pyplot as plt

#Q2.1.5
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')

count = []
index = []
iter = 36
for i in range(36):
    print(i + 1)
	#Rotate Image
    cv_cover_rot = rotate(cv_cover, 10*(i+1), reshape=True)

	#Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(cv_cover, cv_cover_rot)
    if np.size(matches, axis=0) > 1:
        print((i+1)*10)
        plotMatches(cv_cover, cv_cover_rot, matches, locs1, locs2);

	#Update histogram
    for j in range(len(matches)):
        count.append((i+1)*10)
    index.append((i+1)*10)

#Display histogram
plt.hist(count, bins=index, facecolor='blue')
plt.xlabel('angle of rotation',labelpad=10)
plt.ylabel('the number of matches',labelpad=10)
plt.show()

